#pragma once
#include "../core/vector.hpp"

namespace nova::metrics {

inline double mse(
    const Vector& y_true,
    const Vector& y_pred
) {
    return (y_true.eigen() - y_pred.eigen())
            .array()
            .square()
            .mean();
}

}
