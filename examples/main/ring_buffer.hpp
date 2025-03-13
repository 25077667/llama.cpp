
#ifndef __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_RING_HPP__
#define __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_RING_HPP__
#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
#include <array>
#include <stdexcept>
#include <vector>

// Base class template for common ring buffer functionality
template <typename T> struct ring_buffer_base {
    // Returns the first element in the buffer.
    constexpr T & front() {
        if (empty()) {
            throw std::runtime_error("ring buffer is empty");
        }
        return get_data()[get_first()];
    }

    constexpr const T & front() const {
        if (empty()) {
            throw std::runtime_error("ring buffer is empty");
        }
        return get_data()[get_first()];
    }

    // Returns the most recently inserted element.
    constexpr T & back() {
        if (empty()) {
            throw std::runtime_error("ring buffer is empty");
        }
        // Because pos is the next insertion point, the last element is just before pos.
        return get_data()[(get_pos() + get_capacity() - 1) % get_capacity()];
    }

    constexpr const T & back() const {
        if (empty()) {
            throw std::runtime_error("ring buffer is empty");
        }
        return get_data()[(get_pos() + get_capacity() - 1) % get_capacity()];
    }

    // Returns the reverse access element, i.e., the i-th most recently added element.
    constexpr const T & rat(std::size_t i) const {
        if (i >= size()) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return get_data()[(get_first() + size() - i - 1) % get_capacity()];
    }

    // Clears the buffer.
    constexpr void clear() noexcept {
        set_size(0);
        set_first(0);
        set_pos(0);
    }

    constexpr bool empty() const noexcept { return size() == 0; }

  protected:
    // Interface to be implemented by derived classes
    virtual T *         get_data()                   = 0;
    virtual const T *   get_data() const             = 0;
    virtual std::size_t get_capacity() const         = 0;
    virtual std::size_t get_first() const            = 0;
    virtual std::size_t get_pos() const              = 0;
    virtual std::size_t size() const                 = 0;
    virtual void        set_size(std::size_t sz)     = 0;
    virtual void        set_first(std::size_t first) = 0;
    virtual void        set_pos(std::size_t pos)     = 0;
};

// Fixed-size ring buffer using std::array
template <typename T, std::size_t Capacity> struct fixed_ring_buffer : public ring_buffer_base<T> {
    static_assert(Capacity > 0, "Capacity must be greater than zero");

    // default constructor; all members are initialized.
    constexpr fixed_ring_buffer() noexcept : data{}, sz(0), first(0), pos(0) {}

    // Inserts an element at the back of the ring buffer.
    constexpr void push_back(const T & value) {
        // If the buffer is full, the oldest element is overwritten.
        if (sz == Capacity) {
            first = (first + 1) % Capacity;
        } else {
            ++sz;
        }
        data[pos] = value;
        pos       = (pos + 1) % Capacity;
    }

    // Removes and returns the front element.
    constexpr T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first   = (first + 1) % Capacity;
        --sz;
        return value;
    }

    // Converts the ring buffer into a vector.
    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (std::size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % Capacity]);
        }
        return result;
    }

    constexpr std::size_t size() const noexcept override { return sz; }

    constexpr std::size_t capacity() const noexcept { return Capacity; }

  protected:
    T * get_data() override { return data.data(); }

    const T * get_data() const override { return data.data(); }

    std::size_t get_capacity() const override { return Capacity; }

    std::size_t get_first() const override { return first; }

    std::size_t get_pos() const override { return pos; }

    void set_size(std::size_t new_sz) override { sz = new_sz; }

    void set_first(std::size_t new_first) override { first = new_first; }

    void set_pos(std::size_t new_pos) override { pos = new_pos; }

  private:
    std::array<T, Capacity> data;
    std::size_t             sz    = 0;  // number of elements in the buffer
    std::size_t             first = 0;  // index of the first (oldest) element
    std::size_t             pos   = 0;  // next insertion position
};

// Dynamic-size ring buffer using std::vector
template <typename T> struct dynamic_ring_buffer : public ring_buffer_base<T> {
    // Constructor with capacity
    explicit dynamic_ring_buffer(std::size_t capacity) : data(capacity), capacity(capacity), sz(0), first(0), pos(0) {
        if (capacity == 0) {
            throw std::invalid_argument("Capacity must be greater than zero");
        }
    }

    dynamic_ring_buffer() : dynamic_ring_buffer(0) {}

    // Inserts an element at the back of the ring buffer.
    void push_back(const T & value) {
        // If the buffer is full, the oldest element is overwritten.
        if (sz == capacity) {
            first = (first + 1) % capacity;
        } else {
            ++sz;
        }
        data[pos] = value;
        pos       = (pos + 1) % capacity;
    }

    // Removes and returns the front element.
    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first   = (first + 1) % capacity;
        --sz;
        return value;
    }

    // Converts the ring buffer into a vector.
    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (std::size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    std::size_t size() const noexcept override { return sz; }

    std::size_t get_capacity() const override { return capacity; }

  protected:
    T * get_data() override { return data.data(); }

    const T * get_data() const override { return data.data(); }

    std::size_t get_first() const override { return first; }

    std::size_t get_pos() const override { return pos; }

    void set_size(std::size_t new_sz) override { sz = new_sz; }

    void set_first(std::size_t new_first) override { first = new_first; }

    void set_pos(std::size_t new_pos) override { pos = new_pos; }

  private:
    std::vector<T> data;
    std::size_t    capacity;   // fixed capacity of the buffer
    std::size_t    sz    = 0;  // number of elements in the buffer
    std::size_t    first = 0;  // index of the first (oldest) element
    std::size_t    pos   = 0;  // next insertion position
};

// Legacy alias for backward compatibility
template <typename T, std::size_t Capacity> using ring_buffer = fixed_ring_buffer<T, Capacity>;

#endif  // __SCC_EXAMPLE_MAIN_EXPR_TEMP_SAMPLER_RING_HPP__
