#pragma once
#include "../core/vector.hpp"

namespace nova::metrics {

inline double accuracy(
    const Vector& y_true,
    const Vector& y_pred
) {
    int correct = 0;

    for(size_t i=0;i<y_true.size();++i) {
        if(y_true(i) == y_pred(i))
            correct++;
    }

    return double(correct) / y_true.size();
}

}
