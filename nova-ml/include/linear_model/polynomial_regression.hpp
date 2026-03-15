#pragma once
#include <cmath>
#include "../base/estimator.hpp"
#include "linear_regression.hpp"

namespace nova {

class PolynomialRegression : public Estimator {
public:

    explicit PolynomialRegression(int degree)
        : degree_(degree) {}

    void fit(
        const Matrix& X,
        const Vector& y
    ) override {

        Matrix X_poly = transform(X);
        model_.fit(X_poly, y);
    }

    Vector predict(
        const Matrix& X
    ) const override {

        Matrix X_poly = transform(X);
        return model_.predict(X_poly);
    }

private:

    int degree_;
    LinearRegression model_;

    Matrix transform(const Matrix& X) const {

        if (X.cols() != 1)
            throw std::runtime_error(
                "PolynomialRegression supports 1 feature only"
            );

        size_t n = X.rows();

        Matrix X_poly(n, degree_ + 1);

        for (size_t i = 0; i < n; ++i) {

            double val = X.eigen()(i,0);

            X_poly(i,0) = 1.0; // bias term

            for(int d = 1; d <= degree_; ++d) {
                X_poly(i,d) = std::pow(val,d);
            }
        }

        return X_poly;
    }
};

}
