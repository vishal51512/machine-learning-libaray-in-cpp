#pragma once
#include <Eigen/Dense>

namespace nova {

class Vector {
public:
    Vector() = default;

    Vector(size_t size)
        : data_(size) {}

    Vector(const Eigen::VectorXd& v)
        : data_(v) {}

    double& operator()(size_t i) {
        return data_(i);
    }

    const double& operator()(size_t i) const {
        return data_(i);
    }

    size_t size() const { return data_.size(); }

    Eigen::VectorXd& eigen() { return data_; }
    const Eigen::VectorXd& eigen() const { return data_; }

private:
    Eigen::VectorXd data_;
};

}
