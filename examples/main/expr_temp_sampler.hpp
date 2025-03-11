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
#include <ranges>
#include <tuple>
#include <vector>

#include "llama.h"

// Enum defining different sampler types
enum class SamplerType {
    NONE        = 0,
    DRY         = 1,
    TOP_K       = 2,
    TOP_P       = 3,
    MIN_P       = 4,
    TYPICAL_P   = 6,
    TEMPERATURE = 7,
    XTC         = 8,
    INFILL      = 9,
    PENALTIES   = 10,
};

// A common type alias for token probabilities.
using TokenProbs = std::vector<std::pair<int, float>>;

//------------------------------------------------------------------------------
// Polymorphic interface for runtime sampler units.
// (Used for dynamic allocation if needed.)
//------------------------------------------------------------------------------
struct ISamplerUnit {
    virtual constexpr TokenProbs          sample(const TokenProbs & logits) const = 0;
    virtual constexpr const char *        getName() const                         = 0;
    virtual constexpr void                accept(llama_token token)               = 0;
    virtual constexpr void                apply(llama_token_data_array * cur_p)   = 0;
    virtual constexpr void                reset(llama_sampler * smpl)             = 0;
    virtual std::unique_ptr<ISamplerUnit> clone() const                           = 0;
    virtual constexpr void                free()                                  = 0;
    virtual ~ISamplerUnit()                                                       = default;
};

//------------------------------------------------------------------------------
// Base class template to share default (empty) implementations using CRTP.
// We also add a non-virtual operator() to allow our sampler units to be used
// as function objects (required for expression templates).
//------------------------------------------------------------------------------
template <typename Derived> struct SamplerBase : public ISamplerUnit {
    constexpr void accept(llama_token /*token*/) override { /* no-op */ }

    constexpr void apply(llama_token_data_array * /*cur_p*/) override { /* no-op */ }

    constexpr void reset(llama_sampler * /*smpl*/) override { /* no-op */ }

    constexpr void free() override { /* no-op */ }

    std::unique_ptr<ISamplerUnit> clone() const override {
        return std::make_unique<Derived>(static_cast<const Derived &>(*this));
    }

    // Non-virtual call operator for expression template use.
    constexpr TokenProbs operator()(const TokenProbs & logits) const {
        return static_cast<const Derived &>(*this).sample(logits);
    }
};

//------------------------------------------------------------------------------
// SamplerUnit Specializations
//------------------------------------------------------------------------------
template <SamplerType T, typename Derived = void>
class SamplerUnit;  // Primary template not defined; only specializations below.

// SamplerType::NONE: no modification to logits.
template <> class SamplerUnit<SamplerType::NONE> : public SamplerBase<SamplerUnit<SamplerType::NONE>> {
  public:
    constexpr SamplerUnit() noexcept = default;

    constexpr TokenProbs sample(const TokenProbs & logits) const override { return normalize(logits); }

    constexpr const char * getName() const override { return "None"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::DRY: scales logits by a dryness factor.
template <> class SamplerUnit<SamplerType::DRY> : public SamplerBase<SamplerUnit<SamplerType::DRY>> {
    float dryness_;
  public:
    explicit constexpr SamplerUnit(float dryness) noexcept : dryness_(dryness) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs modified = logits;
        for (auto & [token, prob] : modified) {
            prob *= dryness_;
        }
        return normalize(modified);
    }

    constexpr const char * getName() const override { return "Dry"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::TOP_K: selects the top K tokens.
template <> class SamplerUnit<SamplerType::TOP_K> : public SamplerBase<SamplerUnit<SamplerType::TOP_K>> {
    size_t k_;
  public:
    explicit constexpr SamplerUnit(size_t k) noexcept : k_(k) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs sorted_logits = logits;
        std::ranges::sort(sorted_logits, [](const auto & a, const auto & b) { return a.second > b.second; });
        if (k_ < sorted_logits.size()) {
            sorted_logits.resize(k_);
        }
        return normalize(sorted_logits);
    }

    constexpr const char * getName() const override { return "Top-K"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::TOP_P: selects tokens until cumulative probability reaches top_p.
template <> class SamplerUnit<SamplerType::TOP_P> : public SamplerBase<SamplerUnit<SamplerType::TOP_P>> {
    float top_p_;
  public:
    explicit constexpr SamplerUnit(float top_p) noexcept : top_p_(top_p) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs sorted_logits = logits;
        std::ranges::sort(sorted_logits, [](const auto & a, const auto & b) { return a.second > b.second; });
        TokenProbs filtered;
        float      cumulative = 0.0f;
        for (const auto & p : sorted_logits) {
            cumulative += p.second;
            filtered.push_back(p);
            if (cumulative >= top_p_) {
                break;
            }
        }
        return normalize(filtered);
    }

    constexpr const char * getName() const override { return "Top-P"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::MIN_P: filters out tokens below a minimum probability.
template <> class SamplerUnit<SamplerType::MIN_P> : public SamplerBase<SamplerUnit<SamplerType::MIN_P>> {
    float min_p_;
  public:
    explicit constexpr SamplerUnit(float min_p) noexcept : min_p_(min_p) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs filtered;
        for (const auto & p : logits) {
            if (p.second >= min_p_) {
                filtered.push_back(p);
            }
        }
        if (filtered.empty()) {
            return normalize(logits);
        }
        return normalize(filtered);
    }

    constexpr const char * getName() const override { return "Min-P"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::TYPICAL_P: filters tokens based on deviation from average.
template <> class SamplerUnit<SamplerType::TYPICAL_P> : public SamplerBase<SamplerUnit<SamplerType::TYPICAL_P>> {
    float typical_p_;
  public:
    explicit constexpr SamplerUnit(float typical_p) noexcept : typical_p_(typical_p) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        const float avg = sum / static_cast<float>(logits.size());
        TokenProbs  filtered;
        for (const auto & p : logits) {
            if (std::abs(p.second - avg) <= typical_p_) {
                filtered.push_back(p);
            }
        }
        if (filtered.empty()) {
            return normalize(logits);
        }
        return normalize(filtered);
    }

    constexpr const char * getName() const override { return "Typical-P"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & p : logits) {
            norm.emplace_back(p.first, p.second / sum);
        }
        return norm;
    }
};

// SamplerType::TEMPERATURE: adjusts probabilities using a temperature factor.
template <> class SamplerUnit<SamplerType::TEMPERATURE> : public SamplerBase<SamplerUnit<SamplerType::TEMPERATURE>> {
    float temperature_;
  public:
    explicit constexpr SamplerUnit(float temp) noexcept : temperature_(temp) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs modified = logits;
        for (auto & [token, prob] : modified) {
            prob = std::pow(prob, 1.0f / temperature_);
        }
        return normalize(modified);
    }

    constexpr const char * getName() const override { return "Temperature"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::XTC: multiplies probabilities by an XTC parameter.
template <> class SamplerUnit<SamplerType::XTC> : public SamplerBase<SamplerUnit<SamplerType::XTC>> {
    float xtc_param_;
  public:
    explicit constexpr SamplerUnit(float xtc_param) noexcept : xtc_param_(xtc_param) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs modified = logits;
        for (auto & [token, prob] : modified) {
            prob *= xtc_param_;
        }
        return normalize(modified);
    }

    constexpr const char * getName() const override { return "XTC"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::INFILL: boosts probabilities for specific tokens.
// (For full constexpr support, one might use a fixed-size container instead of std::vector.)
template <> class SamplerUnit<SamplerType::INFILL> : public SamplerBase<SamplerUnit<SamplerType::INFILL>> {
    std::vector<int> infill_tokens_;
  public:
    explicit constexpr SamplerUnit(const std::vector<int> & infill_tokens) : infill_tokens_(infill_tokens) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs modified = logits;
        for (auto & [token, prob] : modified) {
            if (std::ranges::find(infill_tokens_, token) != infill_tokens_.end()) {
                prob *= 1.5f;
            }
        }
        return normalize(modified);
    }

    constexpr const char * getName() const override { return "Infill"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & [token, prob] : logits) {
            norm.emplace_back(token, prob / sum);
        }
        return norm;
    }
};

// SamplerType::PENALTIES: subtracts a fixed penalty from each probability.
template <> class SamplerUnit<SamplerType::PENALTIES> : public SamplerBase<SamplerUnit<SamplerType::PENALTIES>> {
    float penalty_;
  public:
    explicit constexpr SamplerUnit(float penalty) noexcept : penalty_(penalty) {}

    constexpr TokenProbs sample(const TokenProbs & logits) const override {
        TokenProbs modified = logits;
        for (auto & [token, prob] : modified) {
            prob = std::max(0.0f, prob - penalty_);
        }
        return normalize(modified);
    }

    constexpr const char * getName() const override { return "Penalties"; }

  private:
    static constexpr TokenProbs normalize(const TokenProbs & logits) {
        const float sum = std::accumulate(logits.begin(), logits.end(), 0.0f,
                                          [](float acc, const auto & p) { return acc + p.second; });
        TokenProbs  norm;
        norm.reserve(logits.size());
        for (const auto & p : logits) {
            norm.emplace_back(p.first, (sum > 0) ? p.second / sum : 0.0f);
        }
        return norm;
    }
};

//------------------------------------------------------------------------------
// Expression Template Components
//------------------------------------------------------------------------------

// SamplerCompose is our expression template that composes two sampler expressions.
template <typename LHS, typename RHS> struct SamplerCompose {
    LHS lhs;
    RHS rhs;

    constexpr TokenProbs operator()(const TokenProbs & logits) const {
        // Apply left-hand sampler expression first, then right-hand.
        return rhs(lhs(logits));
    }
};

// Overload operator| to compose sampler expressions.
template <typename LHS, typename RHS> constexpr auto operator|(const LHS & lhs, const RHS & rhs) {
    return SamplerCompose<LHS, RHS>{ lhs, rhs };
}

// With our SamplerBase providing operator(), each concrete SamplerUnit
// can be used directly as a sampler expression.

//------------------------------------------------------------------------------
// Runtime SamplerChain: (for dynamic polymorphism, as before)
//------------------------------------------------------------------------------
class SamplerChain {
    std::vector<std::unique_ptr<ISamplerUnit>> units_;
  public:
    SamplerChain() = default;

    void add_unit(std::unique_ptr<ISamplerUnit> unit) { units_.push_back(std::move(unit)); }

    TokenProbs sample(const TokenProbs & logits) const {
        TokenProbs result = logits;
        for (const auto & unit : units_) {
            result = unit->sample(result);
        }
        return result;
    }

    void print_chain() const {
        std::cout << "Sampler Chain Units:\n";
        for (const auto & unit : units_) {
            std::cout << " - " << unit->getName() << "\n";
        }
    }
};

#endif  // __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_HPP__
