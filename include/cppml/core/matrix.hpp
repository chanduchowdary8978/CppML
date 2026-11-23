#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

namespace cppml {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, float value);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    float* data() noexcept;
    const float* data() const noexcept;

    // Fast unchecked access in row-major layout.
    float& operator()(std::size_t row, std::size_t col) noexcept;
    const float& operator()(std::size_t row, std::size_t col) const noexcept;

    // Bounds-checked access.
    float& at(std::size_t row, std::size_t col);
    const float& at(std::size_t row, std::size_t col) const;

    void fill(float value);

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;

    [[nodiscard]] std::size_t index(
        std::size_t row,
        std::size_t col
    ) const noexcept;
};

} // namespace cppml