#pragma once
#include "../core/matrix.hpp"
#include "../core/vector.hpp"

namespace nova {

class Estimator {
public:
    virtual ~Estimator() = default;

    virtual void fit(
        const Matrix& X,
        const Vector& y
    ) = 0;

    virtual Vector predict(
        const Matrix& X
    ) const = 0;
};

}
