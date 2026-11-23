
#include "cppml/core/vector.hpp"

#include <algorithm>

namespace cppml {

Vector::Vector(std::size_t size)
    : data_(size) {}

Vector::Vector(std::size_t size, float value)
    : data_(size, value) {}

Vector::Vector(std::initializer_list<float> values)
    : data_(values) {}

std::size_t Vector::size() const noexcept {
    return data_.size();
}

bool Vector::empty() const noexcept {
    return data_.empty();
}

float* Vector::data() noexcept {
    return data_.data();
}

const float* Vector::data() const noexcept {
    return data_.data();
}

float& Vector::operator[](std::size_t index) noexcept {
    assert(index < data_.size());
    return data_[index];
}

const float& Vector::operator[](std::size_t index) const noexcept {
    assert(index < data_.size());
    return data_[index];
}

float& Vector::at(std::size_t index) {
    return data_.at(index);
}

const float& Vector::at(std::size_t index) const {
    return data_.at(index);
}

void Vector::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

} // namespace cppml
