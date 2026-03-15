#pragma once
#include "../base/estimator.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace nova {

class KNNClassifier : public Estimator {
public:

    explicit KNNClassifier(int k = 3)
        : k_(k) {}

    // ===== TRAIN =====
    // KNN just stores the data
    void fit(
        const Matrix& X,
        const Vector& y
    ) override {

        X_train_ = X;
        y_train_ = y;
    }

    // ===== PREDICT =====
    Vector predict(
        const Matrix& X
    ) const override {

        Vector preds(X.rows());

        for(size_t i=0; i<X.rows(); ++i) {

            std::vector<std::pair<double,double>> dist;

            // distance to all training points
            for(size_t j=0; j<X_train_.rows(); ++j) {

                double d = distance(
                    X.eigen().row(i),
                    X_train_.eigen().row(j)
                );

                dist.push_back({d, y_train_(j)});
            }

            // sort by distance
            std::sort(dist.begin(), dist.end(),
                [](auto& a, auto& b){
                    return a.first < b.first;
                });

            // majority vote
            int count0 = 0;
            int count1 = 0;

            for(int t=0; t<k_; ++t) {
                if(dist[t].second == 0)
                    count0++;
                else
                    count1++;
            }

            preds(i) = (count1 > count0) ? 1 : 0;
        }

        return preds;
    }

private:

    int k_;

    Matrix X_train_;
    Vector y_train_;

    double distance(
        const Eigen::VectorXd& a,
        const Eigen::VectorXd& b
    ) const {

        return (a - b).norm(); // Euclidean distance
    }
};

}
