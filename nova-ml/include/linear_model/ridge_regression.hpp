#pragma once
#include "../core/matrix.hpp"
#include "../core/vector.hpp"

namespace nova {

class RidgeRegression {
public:

    RidgeRegression(double alpha = 1.0)
        : alpha_(alpha) {}

    void fit(const Matrix& X, const Vector& y) {

        auto Xt = X.eigen().transpose();

        Eigen::MatrixXd I =
            Eigen::MatrixXd::Identity(
                X.cols(), X.cols());

        coef_ =
            (Xt * X.eigen() + alpha_ * I)
                .ldlt()
                .solve(Xt * y.eigen());
    }

    Vector predict(const Matrix& X) const {
        return Vector(X.eigen() * coef_);
    }

private:
    double alpha_;
    Eigen::VectorXd coef_;
};

}
