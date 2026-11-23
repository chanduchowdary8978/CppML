
#pragma once

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace cppml {

class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t size);
    Vector(std::size_t size, float value);
    Vector(std::initializer_list<float> values);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    float* data() noexcept;
    const float* data() const noexcept;

    float& operator[](std::size_t index) noexcept;
    const float& operator[](std::size_t index) const noexcept;

    float& at(std::size_t index);
    const float& at(std::size_t index) const;

    void fill(float value);

private:
    std::vector<float> data_;
};

} // namespace cppml
