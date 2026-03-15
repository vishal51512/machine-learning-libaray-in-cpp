#pragma once
#include "../core/matrix.hpp"
#include "../core/vector.hpp"
#include "matplotlibcpp.h"

namespace nova::viz {

namespace plt = matplotlibcpp;

// ================= SCATTER =================
inline void scatter_plot(
    const Matrix& X,
    const Vector& y
) {
    std::vector<double> x0, y0;
    std::vector<double> x1, y1;

    for(size_t i=0;i<X.rows();i++) {
        if(y(i)==0) {
            x0.push_back(X.eigen()(i,0));
            y0.push_back(0);
        } else {
            x1.push_back(X.eigen()(i,0));
            y1.push_back(1);
        }
    }

    plt::scatter(x0, y0, 30.0);
    plt::scatter(x1, y1, 30.0);
    plt::save("scatter.png");
    plt::clf();
}

// =============== DECISION BOUNDARY =================
template<typename Model>
inline void decision_boundary(
    Model& model,
    double xmin,
    double xmax
) {
    std::vector<double> xs;
    std::vector<double> ys;

    for(double x=xmin; x<=xmax; x+=0.01) {

        Matrix sample(1,1);
        sample(0,0) = x;

        double pred =
            model.predict(sample)(0);

        xs.push_back(x);
        ys.push_back(pred);
    }

    plt::plot(xs, ys);
    plt::save("decision.png");
    plt::clf();
}

}
