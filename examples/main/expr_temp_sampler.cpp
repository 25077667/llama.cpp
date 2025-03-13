#include <algorithm>
#include <cfloat>
#include <expr_temp_sampler.hpp>
#include <unordered_map>
#include <vector>
#include <cstring>

#include "llama-vocab.h"
#include "llama.h"
#include "ring_buffer.hpp"

static void get_overlapping_token_sequences(
    const llama_vocab & vocab, const std::string_view & str,
    std::unordered_multimap<llama_token, std::vector<llama_token>> & token_sequences, int max_tail_len = -1) {
    for (llama_token token_id = 0; token_id < (llama_token) vocab.n_tokens(); token_id++) {
        std::string word = vocab.detokenize({ token_id }, true);
        if (word.find(str) != std::string::npos) {
            token_sequences.emplace(token_id, std::vector<llama_token>());
        } else {
            size_t word_len = word.size();
            size_t str_len  = str.size();
            size_t pos      = -1;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool   match = true;
                size_t i;
                for (i = 1; i < str_len && i + pos < word_len; ++i) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    std::vector<llama_token> tokenization = vocab.tokenize(str.substr(0, i), false, false);
                    if (max_tail_len >= 0 && tokenization.size() > (size_t) max_tail_len) {
                        tokenization.resize(max_tail_len);
                    }

                    // Ensure we don't already have a duplicate matching tokenization
                    auto its   = token_sequences.equal_range(token_id);
                    bool found = false;
                    for (auto it = its.first; it != its.second; ++it) {
                        if (tokenization == it->second) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        token_sequences.emplace(token_id, tokenization);
                    }
                }
            }
        }
    }
}

void SamplerUnit<SamplerType::DRY>::apply_impl(llama_token_data_array * cur_p) {
    if (!is_dry_enabled()) {
        return;
    }

    int32_t effective_dry_penalty_last_n =
        (dry_penalty_last_n == -1) ? total_context_size : std::max(dry_penalty_last_n, 0);
    int last_n_repeat = std::min(std::min((int) last_tokens.size(), effective_dry_penalty_last_n), total_context_size);

    if (last_n_repeat <= dry_allowed_length) {
        return;
    }

    dry_repeat_count.assign(last_n_repeat, 0);
    dry_max_token_repeat.clear();

    // Step 1: Look for restart sequences to limit the maximum repetition length.
    // Work backwards through the context looking for any token that begins a restart sequence.
    //
    // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
    // sequences that together comprise a restart sequence. This allows us to quickly check
    // whether each token is the head of a complete sequence. Most restart sequences are actually
    // a single token, and for these the "tail" is an empty vector.
    //
    // If the token is a "head", test all restart sequences that begin with this token
    // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
    // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
    // longest matching sequence (if any) is used to limit the maximum repetition length.
    //
    // Note that in the case case of a short sequence contained in a longer one, this might fail to
    // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
    // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
    // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
    //
    // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
    // have already clamped the maximum tail sequence length when generating `restart_sequences`.
    // With clamping, this scan is O(N) in the context length.

    int rep_limit = last_n_repeat;
    for (int i = 0; i < last_n_repeat; ++i) {
        llama_token token = last_tokens.rat(i);
        auto        its   = dry_processed_breakers.equal_range(token);
        if (its.first == dry_processed_breakers.end()) {
            continue;
        }
        int longest_match = -1;
        for (auto it = its.first; it != its.second; ++it) {
            // Note that (*it) does not contain the head character, so seq_len will be
            // the restart sequence length minus 1.
            // In the common case of a single-token restart sequence, (*it) will be empty
            // and we will trivially match.
            int seq_len = (int) it->second.size();
            if (seq_len > longest_match && seq_len <= (int) i) {
                bool match = true;
                for (int offset = 0; offset < seq_len; ++offset) {
                    // The -1 when indexing `last_tokens` is because we already matched the head.
                    if (it->second[offset] != last_tokens.rat(i - offset - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    longest_match = seq_len;
                }
            }
        }
        if (longest_match >= 0) {
            // We found a restart sequence starting `i` tokens from the end and continuing for
            // `longest_match` tokens.
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < dry_allowed_length) {
        return;
    }

    // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
    // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
    // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
    //
    // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
    // https://ivanyu.me/blog/2014/10/15/z-algorithm/
    //
    // The code below is adapted from the public domain implementation by the same author here:
    // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
    //
    // Example:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //                    ^
    //   This `3` means that the last three tokens of the context (a b c) also appear here.
    //
    // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
    // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
    // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
    // ensure that the inner while loops only examine each token in the context once as the outer
    // for loop iterates over the context.

    {
        const int last = last_n_repeat - 1;
        int       rt = 0, lt = 0;

        for (int k = 1; k < last_n_repeat; ++k) {
            if (k > rt) {
                // If k is outside the current Z-box, do naive computation.
                int n = 0;
                while (n + k < last_n_repeat && last_tokens.rat(n) == last_tokens.rat(n + k)) {
                    ++n;
                }
                dry_repeat_count[last - k] = std::min(n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            } else {
                // If k is inside the current Z-box, consider two cases.

                int p              = k - lt;  // Pair index.
                int right_part_len = rt - k + 1;

                if (dry_repeat_count[last - p] < right_part_len) {
                    int n                      = std::min(dry_repeat_count[last - p], rep_limit);
                    dry_repeat_count[last - k] = n;
                } else {
                    int i = rt + 1;
                    while (i < last_n_repeat && last_tokens.rat(i) == last_tokens.rat(i - k)) {
                        i += 1;
                    }

                    int n                      = std::min(i - k, rep_limit);
                    dry_repeat_count[last - k] = n;
                    lt                         = k;
                    rt                         = i - 1;
                }
            }
        }
    }

    // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
    // that would be generated by emitting each new token that would extend a sequence.
    //
    // Following the same example as above:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //
    // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
    // c: 3 -> 4 (from `a b c` to `a b c c`)
    // b: 1 -> 2 (from `c` to `c b`)
    // y: 2 -> 3 (from `b c` to `b c y`)

    for (int i = 0; i < last_n_repeat - 1; ++i) {
        int repeat_len = dry_repeat_count[i];
        if (repeat_len >= dry_allowed_length) {
            // This token ends a repeat, so the next token would continue one.
            // By convention, the value of `repeat_len` only includes the tokens currently
            // in the context, not the new token that would be added.
            llama_token  token = last_tokens.rat(last_n_repeat - 2 - i);
            // Track the maximum sequence ending in this token.
            const auto & it    = dry_max_token_repeat.find(token);
            if (it == dry_max_token_repeat.end() || it->second < repeat_len) {
                dry_max_token_repeat[token] = repeat_len;
            }
        }
    }

    // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

    // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
    // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
    const float FLOAT_MAX_LOG = 88.7228391f;
    int         max_exponent  = 0;
    if (dry_base > 1.000001f) {
        max_exponent = FLOAT_MAX_LOG / std::log(dry_base);
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto & af_kvp = dry_max_token_repeat.find(cur_p->data[i].id);
        if (af_kvp != dry_max_token_repeat.end()) {
            // Check all sequence breakers starting with this token
            auto range                   = dry_processed_breakers.equal_range(cur_p->data[i].id);
            bool is_single_token_breaker = false;

            for (auto it = range.first; it != range.second; ++it) {
                if (it->second.empty()) {
                    is_single_token_breaker = true;
                    break;
                }
            }

            // Apply penalty only if it's not a single-token sequence breaker
            if (!is_single_token_breaker) {
                int repeat_exp = af_kvp->second - dry_allowed_length;
                if (max_exponent > 0 && repeat_exp > max_exponent) {
                    repeat_exp = max_exponent;
                }
                float penalty = dry_multiplier * std::pow(dry_base, repeat_exp);
                cur_p->data[i].logit -= penalty;
            }
        }
    }

    cur_p->sorted = false;
}

// Constructor for DRY SamplerUnit
SamplerUnit<SamplerType::DRY>::SamplerUnit(const llama_vocab * vocab, int32_t context_size, float dry_multiplier,
                                           float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n,
                                           std::span<const std::string_view> seq_breakers) :
    dry_multiplier(dry_multiplier),
    dry_base(dry_base),
    dry_penalty_last_n(dry_penalty_last_n),
    dry_allowed_length(dry_allowed_length),
    total_context_size(context_size) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    constexpr int MAX_CHAR_LEN           = 40;
    constexpr int MAX_SEQ_LEN            = 20;

    const bool dry_enabled = is_dry_enabled();

    if (dry_enabled && !seq_breakers.empty()) {
        // Process sequence breakers
        for (size_t i = 0; i < seq_breakers.size(); ++i) {
            if (seq_breakers[i].empty()) {
                // Skip null or empty sequence breakers
                continue;
            }

            std::string_view sequence_break(seq_breakers[i]);
            if (sequence_break.empty()) {
                continue;
            }

            if (sequence_break.size() > MAX_CHAR_LEN) {
                sequence_break = sequence_break.substr(0, MAX_CHAR_LEN);
            }

            static const auto & llama_vocab_default = llama_vocab();
            const llama_vocab & vocab_ref           = (vocab) ? *vocab : llama_vocab_default;
            get_overlapping_token_sequences(vocab_ref, sequence_break, dry_processed_breakers, MAX_SEQ_LEN);
        }
    }

    if (dry_enabled) {
        dry_repeat_count.resize(effective_dry_penalty_last_n, 0);
        last_tokens = dynamic_ring_buffer<llama_token>(effective_dry_penalty_last_n);
    } else {
        last_tokens = dynamic_ring_buffer<llama_token>(4);
    }
}

constexpr void SamplerUnit<SamplerType::TOP_K>::apply_impl(llama_token_data_array * cur_p) {
    int k = std::min(k_, (uint32_t) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k <= 128) {
            std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size, comp);
        } else {
            constexpr int   nbuckets     = 128;
            constexpr float bucket_low   = -10.0f;
            constexpr float bucket_high  = 10.0f;
            constexpr float bucket_scale = nbuckets / (bucket_high - bucket_low);
            constexpr float bucket_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(cur_p->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int) cur_p->size; ++i) {
                const float val = cur_p->data[i].logit;
                int         ib  = int(bucket_scale * val + bucket_inter);
                ib              = std::max(0, std::min(nbuckets - 1, ib));
                bucket_idx[i]   = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib    = nbuckets - 1;
            for (; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) {
                    break;
                }
            }
            std::vector<llama_token_data>   tmp_tokens(nhave);
            auto *                          ptr = tmp_tokens.data();
            std::vector<llama_token_data *> bucket_ptrs;
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int) cur_p->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets - 1 - j]++ = cur_p->data[i];
                }
            }

            ptr       = tmp_tokens.data();
            int ndone = 0;
            for (int j = nbuckets - 1; j > ib; --j) {
                std::sort(ptr, ptr + histo[j], comp);
                ptr += histo[j];
                ndone += histo[j];
            }
            std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

            std::memcpy(cur_p->data, tmp_tokens.data(), k * sizeof(llama_token_data));
        }
        cur_p->sorted = true;
    }
    cur_p->size = k;
}

void SamplerUnit<SamplerType::PENALTIES>::apply_impl(llama_token_data_array * cur_p) {
    if ((penalty_last_n == 0) || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token_iter = token_count.find(cur_p->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (cur_p->data[i].logit <= 0) {
            cur_p->data[i].logit *= penalty_repeat;
        } else {
            cur_p->data[i].logit /= penalty_repeat;
        }

        cur_p->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    cur_p->sorted = false;
}

void SamplerUnit<SamplerType::PENALTIES>::accept_impl(llama_token token) {
    if (penalty_last_n == 0) {
        return;
    }

    token_count[token]++;

    // if the ring buffer is full, remove the oldest token
    if (prev.size() >= (size_t) penalty_last_n) {
        const auto old = prev.front();

        token_count[old]--;
        if (token_count[old] == 0) {
            token_count.erase(old);
        }
    }

    prev.push_back(token);
}

constexpr static inline void softmax_intl(llama_token_data_array * cur_p) {
    // Sort the logits in descending order
    if (!cur_p->sorted) {
        std::sort(cur_p->data, cur_p->data + cur_p->size,
                  [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
        cur_p->sorted = true;
    }

    float max_l   = cur_p->data[0].logit;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < cur_p->size; ++i) {
        float p          = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= cum_sum;
    }
}

constexpr void SamplerUnit<SamplerType::SOFTMAX>::apply_impl(llama_token_data_array * cur_p) {
    return softmax_intl(cur_p);
}

static inline uint32_t get_rng_seed(uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

static int llama_sample_dist(llama_token_data_array * cur_p, std::mt19937 & rng) {
    // iterator for the probabilities
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

    struct probs_iterator {
        typedef std::input_iterator_tag iterator_category;
        typedef float                   value_type;
        typedef float *                 pointer;
        typedef float &                 reference;
        typedef ptrdiff_t               difference_type;

        const llama_token_data * data;

        bool operator==(const probs_iterator & other) const { return data == other.data; }

        bool operator!=(const probs_iterator & other) const { return data != other.data; }

        const float & operator*() const { return data->p; }

        probs_iterator & operator++() {
            ++data;
            return *this;
        }

        probs_iterator operator++(int) {
            probs_iterator tmp = *this;
            ++data;
            return tmp;
        }
    };

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

    std::discrete_distribution<int> dist(probs_iterator{ cur_p->data }, probs_iterator{ cur_p->data + cur_p->size });

    return dist(rng);
}

inline void SamplerUnit<SamplerType::DIST>::apply_impl(llama_token_data_array * cur_p) {
    softmax_intl(cur_p);
    cur_p->selected = llama_sample_dist(cur_p, rng);
}

SamplerUnit<SamplerType::DIST>::SamplerUnit(uint32_t seed) noexcept : seed(seed), seed_cur(get_rng_seed(seed)) {}

constexpr void SamplerUnit<SamplerType::DIST>::reset() {
    seed_cur = get_rng_seed(seed);
    rng.seed(seed_cur);
}

constexpr void SamplerUnit<SamplerType::TYPICAL_P>::apply_impl(llama_token_data_array * cur_p) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    softmax_intl(cur_p);

    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float shifted_score = fabsf(-logf(cur_p->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(cur_p->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return shifted_scores[a] < shifted_scores[b]; });

    // Compute the cumulative probabilities
    float  cum_sum  = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += cur_p->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> cur_p_new;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        cur_p_new.push_back(cur_p->data[idx]);
    }

    // Replace the data in cur_p with the cur_p_new data
    std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
    cur_p->size   = cur_p_new.size();
    cur_p->sorted = false;
}

constexpr void SamplerUnit<SamplerType::TOP_P>::apply_impl(llama_token_data_array * cur_p) {
    if (p >= 1.0f) {
        return;
    }

    softmax_intl(cur_p);

    // Compute the cumulative probabilities
    float  cum_sum  = 0.0f;
    size_t last_idx = cur_p->size;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    cur_p->size = last_idx;
}

constexpr void SamplerUnit<SamplerType::MIN_P>::apply_impl(llama_token_data_array * cur_p) {
    if (p <= 0.0f || !cur_p->size) {
        return;
    }

    bool min_p_applied = false;

    // if the cur_p aren't sorted, try the unsorted implementation first
    if (!cur_p->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < cur_p->size; ++i) {
            max_logit = std::max(max_logit, cur_p->data[i].logit);
        }
        const float min_logit = max_logit + logf(p);  // min logit for p_i >= p * p_max

        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit >= min_logit) {
                filtered_tokens.push_back(cur_p->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(cur_p->data, filtered_tokens.data(), filtered_tokens.size() * sizeof(llama_token_data));
            cur_p->size   = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the cur_p are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!cur_p->sorted) {
            std::sort(cur_p->data, cur_p->data + cur_p->size,
                      [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
            cur_p->sorted = true;
        }

        const float min_logit = cur_p->data[0].logit + logf(p);  // min logit for p_i >= p * p_max
        size_t      i         = 1;                               // first token always matches

        for (; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit < min_logit && i >= min_keep) {
                break;  // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        cur_p->size = i;
    }
}

constexpr void SamplerUnit<SamplerType::TEMPERATURE>::apply_impl(llama_token_data_array * cur_p) {
    if (temp <= 0.0f) {
        // find the token with the highest logit and set the rest to -inf
        size_t max_i = 0;
        float  max_l = cur_p->data[0].logit;

        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit > max_l) {
                cur_p->data[max_i].logit = -INFINITY;
                max_i                    = i;
                max_l                    = cur_p->data[i].logit;
            } else {
                cur_p->data[i].logit = -INFINITY;
            }
        }

        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}

SamplerUnit<SamplerType::XTC>::SamplerUnit(float p, float t, size_t min_keep, uint32_t seed) noexcept :
    probability(p),
    threshold(t),
    min_keep(min_keep),
    seed(seed),
    seed_cur(get_rng_seed(seed)),
    rng(seed_cur) {}

constexpr void SamplerUnit<SamplerType::XTC>::apply_impl(llama_token_data_array * cur_p) {
    if (probability <= 0.0f || threshold > 0.5f || cur_p->size < 2) {
        return;
    }

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float                                 chance = distribution(rng);
    if (chance > probability) {
        return;
    }

    // in case it's not sorted/recalculated yet
    softmax_intl(cur_p);

    int pos_last = 0;

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p >= threshold) {
            pos_last = i;
        } else {
            break;
        }
    }

    if (cur_p->size - pos_last >= min_keep && pos_last > 0) {
        cur_p->data += pos_last;
        cur_p->size -= pos_last;
    }
}

constexpr void SamplerUnit<SamplerType::XTC>::reset() {
    seed_cur = get_rng_seed(seed);
    rng.seed(seed_cur);
}
