#pragma once
#include "../base/estimator.hpp"
#include "../tree/decision_tree_classifier.hpp"
#include <vector>
#include <random>
#include <memory>

namespace nova {

class RandomForestClassifier : public Estimator {
public:

    RandomForestClassifier(
        int n_trees = 10,
        int max_depth = 3
    )
        : n_trees_(n_trees),
          max_depth_(max_depth) {}

    void fit(const Matrix& X,
             const Vector& y) override {

        trees_.clear();

        std::mt19937 gen(42);
        std::uniform_int_distribution<> dist(0, X.rows() - 1);

        for(int t = 0; t < n_trees_; t++) {

            Matrix X_sample(X.rows(), X.cols());
            Vector y_sample(X.rows());

            for(size_t i = 0; i < X.rows(); i++) {
                int idx = dist(gen);
                X_sample.eigen().row(i) =
                    X.eigen().row(idx);
                y_sample(i) = y(idx);
            }

            auto tree =
                std::make_unique<DecisionTreeClassifier>(max_depth_);

            tree->fit(X_sample, y_sample);

            trees_.push_back(std::move(tree));
        }
    }

    Vector predict(const Matrix& X) const override {

        Vector preds(X.rows());

        for(size_t i = 0; i < X.rows(); i++) {

            int count0 = 0;
            int count1 = 0;

            Matrix sample(1, X.cols());
            sample.eigen().row(0) =
                X.eigen().row(i);

            for(const auto& tree : trees_) {

                double p =
                    tree->predict(sample)(0);

                if(p == 0) count0++;
                else count1++;
            }

            preds(i) =
                (count1 > count0) ? 1 : 0;
        }

        return preds;
    }

private:
    int n_trees_;
    int max_depth_;

    std::vector<std::unique_ptr<DecisionTreeClassifier>> trees_;
};

}
