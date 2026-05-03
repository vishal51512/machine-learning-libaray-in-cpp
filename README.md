# Nova ML (C++)

Lightweight, header-only machine learning utilities in C++ built on top of Eigen.

## Features
- Core data structures: `Matrix` and `Vector` wrappers around `Eigen::MatrixXd`/`Eigen::VectorXd`
- Linear models: Linear Regression, Ridge Regression, Polynomial Regression, Logistic Regression
- Neighbors: KNN Classifier
- Tree models: Decision Tree Classifier, Random Forest Classifier
- Model selection: Train/test split helper
- Metrics: Accuracy and utility metrics
- Visualization helpers via `matplotlibcpp` (optional)

## Requirements
- C++17 compiler
- [Eigen](https://eigen.tuxfamily.org/) headers available on your include path
- (Optional) Python + matplotlib for visualization helpers

## Build and run the example
The sample entry point is `nova-ml/main.cpp`.

```bash
g++ -std=c++17 \
  -I./nova-ml \
  -I./nova-ml/include \
  -I<path-to-eigen-headers> \
  ./nova-ml/main.cpp \
  -o nova-ml-demo

./nova-ml-demo
```

## Project structure
```
nova-ml/
  include/
    base/           Estimator interface
    core/           Matrix/Vector wrappers
    linear_model/   Linear & logistic models
    metrics/        Accuracy and utilities
    model_selection/Train/test split
    neighbors/      KNN classifier
    tree/           Decision tree & random forest
    visualization/  Plot helpers (optional)
  main.cpp          Example usage
```
