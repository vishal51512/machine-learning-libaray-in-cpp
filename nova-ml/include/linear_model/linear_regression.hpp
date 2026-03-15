#pragma once
#include "../base/estimator.hpp"

namespace nova {

class LinearRegression : public Estimator {
public:

    void fit(
        const Matrix& X,
        const Vector& y
    ) override {

        auto Xt = X.eigen().transpose();

        coef_ =
            (Xt * X.eigen())
            .ldlt()
            .solve(Xt * y.eigen());
    }

    Vector predict(
        const Matrix& X
    ) const override {

        return Vector(X.eigen() * coef_);
    }

private:
    Eigen::VectorXd coef_;
};

}
