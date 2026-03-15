#pragma once
#include "../base/estimator.hpp"
#include <vector>
#include <limits>
#include <algorithm>

namespace nova {

class DecisionTreeClassifier : public Estimator {
public:

    explicit DecisionTreeClassifier(int max_depth = 3)
        : max_depth_(max_depth), root_(nullptr) {}

    ~DecisionTreeClassifier() {
        free_tree(root_);
    }

    void fit(const Matrix& X,
             const Vector& y) override {

        root_ = build_tree(X, y, 0);
    }

    Vector predict(const Matrix& X) const override {

        Vector preds(X.rows());

        for(size_t i=0;i<X.rows();i++) {
            preds(i) = predict_sample(
                X.eigen().row(i),
                root_
            );
        }

        return preds;
    }

private:

    struct Node {
        bool is_leaf;
        int feature_index;
        double threshold;
        double value;
        Node* left;
        Node* right;

        Node() :
            is_leaf(false),
            feature_index(-1),
            threshold(0),
            value(0),
            left(nullptr),
            right(nullptr) {}
    };

    Node* root_;
    int max_depth_;

    // ================= BUILD TREE =================

    Node* build_tree(const Matrix& X,
                     const Vector& y,
                     int depth) {

        Node* node = new Node();

        // stopping conditions
        if(depth >= max_depth_ ||
           gini(y) == 0.0 ||
           X.rows() <= 1) {

            node->is_leaf = true;
            node->value = majority_class(y);
            return node;
        }

        int best_feature = -1;
        double best_threshold = 0;
        double best_score = std::numeric_limits<double>::max();

        // try all features
        for(int f=0; f<X.cols(); f++) {

            for(int i=0; i<X.rows(); i++) {

                double threshold = X.eigen()(i,f);

                auto [left_y, right_y] =
                    split_y(X, y, f, threshold);

                if(left_y.size()==0 ||
                   right_y.size()==0)
                    continue;

                double score =
                    weighted_gini(left_y, right_y);

                if(score < best_score) {
                    best_score = score;
                    best_feature = f;
                    best_threshold = threshold;
                }
            }
        }

        if(best_feature == -1) {
            node->is_leaf = true;
            node->value = majority_class(y);
            return node;
        }

        node->feature_index = best_feature;
        node->threshold = best_threshold;

        auto [X_left, y_left,
              X_right, y_right] =
            split_dataset(X, y,
                          best_feature,
                          best_threshold);

        node->left =
            build_tree(X_left, y_left,
                       depth+1);

        node->right =
            build_tree(X_right, y_right,
                       depth+1);

        return node;
    }

    // ================= PREDICT =================

    double predict_sample(
        const Eigen::VectorXd& x,
        Node* node) const {

        if(node->is_leaf)
            return node->value;

        if(x(node->feature_index)
           <= node->threshold)
            return predict_sample(x,
                                  node->left);
        else
            return predict_sample(x,
                                  node->right);
    }

    // ================= HELPERS =================

    double gini(const Vector& y) const {

        int count0=0, count1=0;

        for(size_t i=0;i<y.size();i++) {
            if(y(i)==0) count0++;
            else count1++;
        }

        double p0 =
            double(count0)/y.size();
        double p1 =
            double(count1)/y.size();

        return 1 - (p0*p0 + p1*p1);
    }

    double weighted_gini(
        const Vector& left,
        const Vector& right) const {

        double n = left.size()
                   + right.size();

        return (left.size()/n)
               * gini(left)
             + (right.size()/n)
               * gini(right);
    }

    double majority_class(
        const Vector& y) const {

        int count0=0, count1=0;

        for(size_t i=0;i<y.size();i++) {
            if(y(i)==0) count0++;
            else count1++;
        }

        return (count1 > count0)
               ? 1 : 0;
    }

    std::pair<Vector,Vector>
    split_y(const Matrix& X,
            const Vector& y,
            int feature,
            double threshold) const {

        std::vector<double> left, right;

        for(size_t i=0;i<X.rows();i++) {

            if(X.eigen()(i,feature)
               <= threshold)
                left.push_back(y(i));
            else
                right.push_back(y(i));
        }

        Vector l(left.size()),
               r(right.size());

        for(size_t i=0;i<left.size();i++)
            l(i)=left[i];

        for(size_t i=0;i<right.size();i++)
            r(i)=right[i];

        return {l,r};
    }

    std::tuple<Matrix,Vector,
               Matrix,Vector>
    split_dataset(const Matrix& X,
                  const Vector& y,
                  int feature,
                  double threshold) const {

        std::vector<int> left_idx,
                         right_idx;

        for(size_t i=0;i<X.rows();i++) {

            if(X.eigen()(i,feature)
               <= threshold)
                left_idx.push_back(i);
            else
                right_idx.push_back(i);
        }

        Matrix X_left(left_idx.size(),
                      X.cols());
        Matrix X_right(right_idx.size(),
                       X.cols());

        Vector y_left(left_idx.size());
        Vector y_right(right_idx.size());

        for(size_t i=0;i<left_idx.size();i++){
            X_left.eigen().row(i)=
                X.eigen().row(left_idx[i]);
            y_left(i)=y(left_idx[i]);
        }

        for(size_t i=0;i<right_idx.size();i++){
            X_right.eigen().row(i)=
                X.eigen().row(right_idx[i]);
            y_right(i)=y(right_idx[i]);
        }

        return {X_left,y_left,
                X_right,y_right};
    }

    void free_tree(Node* node) {
        if(!node) return;
        free_tree(node->left);
        free_tree(node->right);
        delete node;
    }
};

}
