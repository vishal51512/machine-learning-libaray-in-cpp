#pragma once
#include <Eigen/Dense>

namespace nova {

class Matrix {
public:
    Matrix() = default;

    Matrix(size_t rows, size_t cols)
        : data_(rows, cols) {}

    double& operator()(size_t i, size_t j) {
        return data_(i, j);
    }

    const double& operator()(size_t i, size_t j) const {
        return data_(i, j);
    }

    size_t rows() const { return data_.rows(); }
    size_t cols() const { return data_.cols(); }

    Eigen::MatrixXd& eigen() { return data_; }
    const Eigen::MatrixXd& eigen() const { return data_; }

private:
    Eigen::MatrixXd data_;
};

}
