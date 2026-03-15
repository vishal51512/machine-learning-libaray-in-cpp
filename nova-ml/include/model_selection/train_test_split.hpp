#pragma once
#include "../core/matrix.hpp"
#include "../core/vector.hpp"
#include <vector>
#include <algorithm>
#include <random>


namespace nova {

struct SplitResult {
    Matrix X_train;
    Matrix X_test;
    Vector y_train;
    Vector y_test;
};

inline SplitResult train_test_split(
    const Matrix& X,
    const Vector& y,
    double test_size = 0.2,
    bool shuffle = true,
    int random_state = 42
) {
    size_t n = X.rows();

    size_t test_count =
        static_cast<size_t>(n * test_size);

    size_t train_count = n - test_count;
    // create index vector 
    std::vector<size_t> indices(n);
    for(size_t i = 0 ; i < n ; i++) indices[i] = i;
    // shuffle if required 
    if(shuffle)
    {
     	std::mt19937 gen(random_state);
     	std::shuffle(indices.begin(),indices.end(),gen);
    }
     

    Matrix X_train(train_count, X.cols());
    Matrix X_test(test_count, X.cols());

    Vector y_train(train_count);
    Vector y_test(test_count);

    for (size_t i = 0; i < train_count; i++) {
        X_train.eigen().row(i) = X.eigen().row(indices[i]);
        y_train.eigen()(i) = y.eigen()(indices[i]);
    }

    for (size_t i = 0; i < test_count; i++) {
        X_test.eigen().row(i) =
            X.eigen().row(indices[train_count + i]);

        y_test.eigen()(i) =
            y.eigen()(indices[train_count + i]);
    }

    return {X_train, X_test, y_train, y_test};
}

}
