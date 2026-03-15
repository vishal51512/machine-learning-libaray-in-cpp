#pragma once
#include "../base/estimator.hpp"
#include <cmath>

namespace nova {

class LogisticRegression : public Estimator {
public:

    LogisticRegression(
        double lr = 0.01,
        int epochs = 1000
    )
        : lr_(lr), epochs_(epochs) {}

    // ===== TRAIN =====
    void fit(
        const Matrix& X,
        const Vector& y
    ) override {

        int n = X.rows();
        int m = X.cols();

        weights_ = Eigen::VectorXd::Zero(m);

        for(int e = 0; e < epochs_; ++e) {

            Eigen::VectorXd preds =
                sigmoid(X.eigen() * weights_);

            Eigen::VectorXd gradient =
                (X.eigen().transpose() *
                (preds - y.eigen())) / n;

            weights_ -= lr_ * gradient;
        }
    }

    // ===== PREDICT LABEL =====
    Vector predict(
        const Matrix& X
    ) const override {

        Eigen::VectorXd probs =
            sigmoid(X.eigen() * weights_);

        Eigen::VectorXd labels =
            (probs.array() > 0.5)
                .cast<double>();

        return Vector(labels);
    }

    // ===== PREDICT PROBABILITY =====
    Vector predict_proba(
        const Matrix& X
    ) const {

        return Vector(
            sigmoid(X.eigen() * weights_)
        );
    }

private:

    double lr_;
    int epochs_;

    Eigen::VectorXd weights_;

    Eigen::VectorXd sigmoid(
        const Eigen::VectorXd& z
    ) const {

        return 1.0 /
            (1.0 + (-z.array()).exp());
    }
};

}
