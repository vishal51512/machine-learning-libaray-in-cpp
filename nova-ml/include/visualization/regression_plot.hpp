#pragma once
#include "../core/matrix.hpp"
#include "../core/vector.hpp"
#include <vector>
#include "plot.hpp"

namespace nova::viz {

inline void regression_plot(
    const Matrix& X,
    const Vector& y,
    const Vector& preds
) {
    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> pred_vals;

    x_vals.reserve(X.rows());
    y_vals.reserve(y.size());
    pred_vals.reserve(preds.size());

    // assumes 1D regression (X has 1 column)
    for(size_t i = 0; i < X.rows(); ++i) {
        x_vals.push_back(X.eigen()(i,0));
        y_vals.push_back(y.eigen()(i));
        pred_vals.push_back(preds.eigen()(i));
    }

    // your plotting backend
    plot::scatter(x_vals, y_vals);   // real data
    plot::line(x_vals, pred_vals);   // regression line
    plot::show();
}

}
