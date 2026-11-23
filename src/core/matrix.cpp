#include "cppml/core/matrix.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cppml {

namespace {

std::size_t checked_size(std::size_t rows, std::size_t cols) {
    if (cols != 0 &&
        rows > std::numeric_limits<std::size_t>::max() / cols) {
        throw std::length_error("Matrix dimensions overflow");
    }

    return rows * cols;
}

} // namespace

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(checked_size(rows, cols)) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, float value)
    : rows_(rows),
      cols_(cols),
      data_(checked_size(rows, cols), value) {}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::cols() const noexcept {
    return cols_;
}

std::size_t Matrix::size() const noexcept {
    return data_.size();
}

bool Matrix::empty() const noexcept {
    return data_.empty();
}

float* Matrix::data() noexcept {
    return data_.data();
}

const float* Matrix::data() const noexcept {
    return data_.data();
}

std::size_t Matrix::index(
    std::size_t row,
    std::size_t col
) const noexcept {
    return row * cols_ + col;
}

float& Matrix::operator()(
    std::size_t row,
    std::size_t col
) noexcept {
    assert(row < rows_ && col < cols_);
    return data_[index(row, col)];
}

const float& Matrix::operator()(
    std::size_t row,
    std::size_t col
) const noexcept {
    assert(row < rows_ && col < cols_);
    return data_[index(row, col)];
}

float& Matrix::at(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_.at(index(row, col));
}

const float& Matrix::at(
    std::size_t row,
    std::size_t col
) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data_.at(index(row, col));
}

void Matrix::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

} // namespace cppml