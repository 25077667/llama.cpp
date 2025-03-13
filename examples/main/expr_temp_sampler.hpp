#ifndef __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_HPP__
#define __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_HPP__
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <ranges>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "llama.h"
#include "ring_buffer.hpp"

//---------------------------------------------------------------------
// Extended SamplerType enum.
enum class SamplerType {
    NONE        = 0,
    DRY         = 1,  // Do not Repeat Yourself
    TOP_K       = 2,
    TOP_P       = 3,
    MIN_P       = 4,
    TYPICAL_P   = 6,
    TEMPERATURE = 7,
    XTC         = 8,
    INFILL      = 9,
    PENALTIES   = 10,
    GRAMMAR     = 11,
    SOFTMAX     = 12,
    DIST        = 13,
    GREEDY      = 14
};

//---------------------------------------------------------------------
// Polymorphic interface for runtime sampler units.
struct ISamplerUnit {
    virtual constexpr const char *        getName() const                       = 0;
    virtual constexpr void                accept(llama_token token)             = 0;
    virtual constexpr void                apply(llama_token_data_array * cur_p) = 0;
    virtual constexpr void                reset()                               = 0;
    virtual std::unique_ptr<ISamplerUnit> clone() const                         = 0;
    virtual constexpr void                free()                                = 0;
    virtual ~ISamplerUnit()                                                     = default;
};

//---------------------------------------------------------------------
// Base class template using CRTP.
// Provides a shared normalize() function.
template <typename Derived> struct SamplerBase : public ISamplerUnit {
    // Default no-op implementations.
    constexpr void accept(llama_token /*token*/) override {}

    constexpr void apply(llama_token_data_array * /*cur_p*/) override {}

    constexpr void reset() override {}

    constexpr void free() override {}

    std::unique_ptr<ISamplerUnit> clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived &>(*this));
    }

    // Non-virtual call operator for expression template use.
    constexpr void apply(llama_token_data_array * cur_p) const { static_cast<const Derived *>(this)->apply(cur_p); }

    // Non-virtual call operator for expression template use.
    constexpr void accept(llama_token token) const { static_cast<const Derived *>(this)->accept(token); }
};

//---------------------------------------------------------------------
// Forward declaration of SamplerUnit specializations.
template <SamplerType T, typename Derived = void> class SamplerUnit;  // Primary template not defined.

//---------------------------------------------------------------------
// Example sampler units:

template <> class SamplerUnit<SamplerType::DRY> : public SamplerBase<SamplerUnit<SamplerType::DRY>> {
    const float   dry_multiplier     = 0.0f;
    const float   dry_base           = 1.75f;
    const int32_t dry_penalty_last_n = -1;
    const int32_t dry_allowed_length = 2;
    int32_t       total_context_size;

    std::unordered_multimap<llama_token, std::vector<llama_token>> dry_processed_breakers;
    std::vector<int>                                               dry_repeat_count;
    std::unordered_map<llama_token, int>                           dry_max_token_repeat;
    dynamic_ring_buffer<llama_token>                               last_tokens;

    void apply_impl(llama_token_data_array *);

    inline bool is_dry_enabled() const { return dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0; }

  public:
    SamplerUnit(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length,
                int32_t dry_penalty_last_n, const char ** seq_breakers, size_t num_breakers);

    constexpr void apply(llama_token_data_array * cur_p) override { apply_impl(cur_p); }

    constexpr void accept(llama_token token) override {
        if (is_dry_enabled()) {
            last_tokens.push_back(token);
        }
    }

    constexpr const char * getName() const override { return "Dry"; }
};

// TOP_K: selects the top K tokens.
template <> class SamplerUnit<SamplerType::TOP_K> : public SamplerBase<SamplerUnit<SamplerType::TOP_K>> {
    const uint32_t k_ = 40;

    constexpr void apply_impl(llama_token_data_array * cur_p);

  public:
    constexpr SamplerUnit(uint32_t k) noexcept : k_(k) {}

    constexpr void apply(llama_token_data_array * cur_p) override { apply_impl(cur_p); }

    constexpr const char * getName() const override { return "Top-K"; }
};

// PENALTIES: subtracts a combined penalty from each probability.
template <> class SamplerUnit<SamplerType::PENALTIES> : public SamplerBase<SamplerUnit<SamplerType::PENALTIES>> {
    const int32_t           penalty_last_n;
    const float             penalty_repeat;
    const float             penalty_freq;
    const float             penalty_present;
    static constexpr size_t ring_buffer_size = 64;  // same as common_params_sampling::penalty_last_n

    // a frequency map to count token occurrences
    std::unordered_map<llama_token, int> token_count;
    void                                 apply_impl(llama_token_data_array * cur_p);
    void                                 accept_impl(llama_token token);

    fixed_ring_buffer<llama_token, ring_buffer_size> prev;

  public:
    SamplerUnit(int32_t last_n, float repeat, float freq, float present) noexcept :
        penalty_last_n(last_n),
        penalty_repeat(repeat),
        penalty_freq(freq),
        penalty_present(present) {}

    void apply(llama_token_data_array * cur_p) override { apply_impl(cur_p); }

    void accept(llama_token token) override { accept_impl(token); }

    constexpr void reset() override {
        prev.clear();
        token_count.clear();
    }

    constexpr const char * getName() const override { return "Penalties"; }
};

// GRAMMAR: dummy unit; if grammar is nullptr, does nothing.
template <> class SamplerUnit<SamplerType::GRAMMAR> : public SamplerBase<SamplerUnit<SamplerType::GRAMMAR>> {
    [[maybe_unused]] void * model_;
    [[maybe_unused]] void * grammar_;
  public:
    // Always initialize with a pointer—even if nullptr.
    constexpr SamplerUnit(void * model, void * grammar) noexcept : model_(model), grammar_(grammar) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Grammar"; }
};

// SOFTMAX: applies a softmax operation.
template <> class SamplerUnit<SamplerType::SOFTMAX> : public SamplerBase<SamplerUnit<SamplerType::SOFTMAX>> {
    constexpr static void apply_impl(llama_token_data_array * cur_p);

  public:
    constexpr SamplerUnit() noexcept {}

    constexpr void apply(llama_token_data_array * cur_p) override { apply_impl(cur_p); }

    constexpr const char * getName() const override { return "Softmax"; }
};

// DIST: dummy distribution stage; selects the token with maximum probability.
template <> class SamplerUnit<SamplerType::DIST> : public SamplerBase<SamplerUnit<SamplerType::DIST>> {
    const uint32_t seed;
    uint32_t       seed_cur;

    std::mt19937 rng;

    void apply_impl(llama_token_data_array * cur_p);

  public:
    SamplerUnit(uint32_t seed) noexcept;

    constexpr void apply(llama_token_data_array * cur_p) override { apply_impl(cur_p); }

    constexpr void reset() override;

    constexpr const char * getName() const override { return "Dist"; }
};

// GREEDY: dummy greedy selection; same as DIST.
template <> class SamplerUnit<SamplerType::GREEDY> : public SamplerBase<SamplerUnit<SamplerType::GREEDY>> {
    [[maybe_unused]] float dummy_param_;
  public:
    SamplerUnit(float param) noexcept : dummy_param_(param) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Greedy"; }
};

// TYPICAL_P: filters tokens based on deviation from average.
template <> class SamplerUnit<SamplerType::TYPICAL_P> : public SamplerBase<SamplerUnit<SamplerType::TYPICAL_P>> {
    float typical_p_;
    int   min_keep_;
  public:
    SamplerUnit(float typical_p, int min_keep) noexcept : typical_p_(typical_p), min_keep_(min_keep) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Typical-P"; }
};

// TOP_P: filters tokens until cumulative probability reaches top_p.
template <> class SamplerUnit<SamplerType::TOP_P> : public SamplerBase<SamplerUnit<SamplerType::TOP_P>> {
    float top_p_;
    int   min_keep_;
  public:
    SamplerUnit(float top_p, int min_keep) noexcept : top_p_(top_p), min_keep_(min_keep) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Top-P"; }
};

// MIN_P: filters out tokens below a minimum probability.
template <> class SamplerUnit<SamplerType::MIN_P> : public SamplerBase<SamplerUnit<SamplerType::MIN_P>> {
    float min_p_;
    int   min_keep_;
  public:
    SamplerUnit(float min_p, int min_keep) noexcept : min_p_(min_p), min_keep_(min_keep) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Min-P"; }
};

// TEMPERATURE: scales probabilities by temperature.
template <> class SamplerUnit<SamplerType::TEMPERATURE> : public SamplerBase<SamplerUnit<SamplerType::TEMPERATURE>> {
    float temperature_;
  public:
    SamplerUnit(float temp) noexcept : temperature_(temp) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Temperature"; }
};

// XTC: multiplies probabilities by an XTC parameter.
template <> class SamplerUnit<SamplerType::XTC> : public SamplerBase<SamplerUnit<SamplerType::XTC>> {
    float xtc_param_;
  public:
    SamplerUnit(float xtc_param) noexcept : xtc_param_(xtc_param) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "XTC"; }
};

// INFILL: boosts probabilities for specific tokens.
template <> class SamplerUnit<SamplerType::INFILL> : public SamplerBase<SamplerUnit<SamplerType::INFILL>> {
    std::vector<int> infill_tokens_;
  public:
    // Note: std::vector is not fully constexpr-friendly; placeholder implementation.
    SamplerUnit(const std::vector<int> & infill_tokens) : infill_tokens_(infill_tokens) {}

    constexpr void apply(llama_token_data_array *) override { ; }

    constexpr const char * getName() const override { return "Infill"; }
};

//---------------------------------------------------------------------
// Expression Template Components:
// SamplerCompose composes two sampler expressions.
template <typename LHS, typename RHS> struct SamplerCompose {
    LHS lhs;
    RHS rhs;

    constexpr void apply(llama_token_data_array * cur_p) const {
        lhs.apply(cur_p);
        rhs.apply(cur_p);
    }

    constexpr void accept(llama_token token) const {
        lhs.accept(token);
        rhs.accept(token);
    }
};

// Overload operator| to compose sampler expressions.
template <typename LHS, typename RHS> constexpr auto operator|(const LHS & lhs, const RHS & rhs) {
    return SamplerCompose<LHS, RHS>{ lhs, rhs };
}

//---------------------------------------------------------------------
// Identity sampler lambda: returns its input unmodified.
auto IdentitySampler = [](auto chain) {
    return chain;
};

//---------------------------------------------------------------------
// Runtime SamplerChain (for dynamic polymorphism).
class SamplerChain {
    std::vector<std::unique_ptr<ISamplerUnit>> units_;
  public:
    SamplerChain() = default;

    void add_unit(std::unique_ptr<ISamplerUnit> unit) { units_.push_back(std::move(unit)); }

    void apply(llama_token_data_array * cur_p) const {
        for (const auto & unit : units_) {
            unit->apply(cur_p);
        }
    }

    void print_chain() const {
        std::cout << "Sampler Chain Units:\n";
        for (const auto & unit : units_) {
            std::cout << " - " << unit->getName() << "\n";
        }
    }
};

//---------------------------------------------------------------------
// Test Cases
//
// Assumed variables:
struct CommonSamplingParams {
    static const int       n_vocab            = 50000;
    static const int       token_eos          = 2;
    static const int       token_nl           = 10;
    static const int       last_n_tokens_size = 64;
    static constexpr float repeat_penalty     = 1.1f;
    static constexpr float frequency_penalty  = 0.5f;
    static constexpr float presence_penalty   = 0.3f;
    static const bool      penalize_nl        = true;
    static const int       seed               = 42;
    static const int       top_k              = 40;
    static constexpr float typical_p          = 0.9f;
    static constexpr float top_p              = 0.8f;
    static constexpr float min_p              = 0.05f;
    static constexpr float temp               = 1.0f;  // For test case 3.
    static const int       min_keep           = 1;     // Default min_keep value.
};

void * model   = nullptr;  // Dummy model pointer.
void * grammar = nullptr;  // Dummy grammar pointer.

//--- 1. Case: temp < 0.0 ---
// Chain: Penalties, Grammar, Softmax, then Distribution.
auto filter_stack_temp_negative =
    SamplerUnit<SamplerType::PENALTIES>(CommonSamplingParams::last_n_tokens_size, CommonSamplingParams::repeat_penalty,
                                        CommonSamplingParams::frequency_penalty,
                                        CommonSamplingParams::presence_penalty) |
    SamplerUnit<SamplerType::GRAMMAR>(model, grammar) | SamplerUnit<SamplerType::SOFTMAX>() |
    SamplerUnit<SamplerType::DIST>(CommonSamplingParams::seed);
//--- 2. Case: temp == 0.0 ---
// Chain: Penalties, Grammar, then Greedy selection.
auto filter_stack_temp_zero =
    SamplerUnit<SamplerType::PENALTIES>(CommonSamplingParams::last_n_tokens_size, CommonSamplingParams::repeat_penalty,
                                        CommonSamplingParams::frequency_penalty,
                                        CommonSamplingParams::presence_penalty) |
    SamplerUnit<SamplerType::GRAMMAR>(model, grammar) | SamplerUnit<SamplerType::GREEDY>(0.0f);

//--- 3. Case: temp > 0.0 ---
// Chain: Penalties, Grammar, Top-K, Typical-P, Top-P, Min-P,
// Temperature scaling, and Distribution.
auto filter_stack_temp_positive =
    SamplerUnit<SamplerType::PENALTIES>(CommonSamplingParams::last_n_tokens_size, CommonSamplingParams::repeat_penalty,
                                        CommonSamplingParams::frequency_penalty,
                                        CommonSamplingParams::presence_penalty) |
    SamplerUnit<SamplerType::GRAMMAR>(model, grammar) | SamplerUnit<SamplerType::TOP_K>(CommonSamplingParams::top_k) |
    SamplerUnit<SamplerType::TYPICAL_P>(CommonSamplingParams::typical_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::TOP_P>(CommonSamplingParams::top_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::MIN_P>(CommonSamplingParams::min_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::TEMPERATURE>(CommonSamplingParams::temp) |
    SamplerUnit<SamplerType::DIST>(CommonSamplingParams::seed);

//--- 4. Case: example common filter stack ---
// Chain: Penalties, DRY, Top-K, Typical-P, Top-P, Min-P, XTC, Temperature
auto filter_stack_example_common =
    SamplerUnit<SamplerType::PENALTIES>(CommonSamplingParams::last_n_tokens_size, CommonSamplingParams::repeat_penalty,
                                        CommonSamplingParams::frequency_penalty,
                                        CommonSamplingParams::presence_penalty) |
    // SamplerUnit<SamplerType::DRY>(0.9f) |
    SamplerUnit<SamplerType::TOP_K>(CommonSamplingParams::top_k) |
    SamplerUnit<SamplerType::TYPICAL_P>(CommonSamplingParams::typical_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::TOP_P>(CommonSamplingParams::top_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::MIN_P>(CommonSamplingParams::min_p, CommonSamplingParams::min_keep) |
    SamplerUnit<SamplerType::XTC>(1.5f) | SamplerUnit<SamplerType::TEMPERATURE>(CommonSamplingParams::temp);

#endif  // __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_HPP__
