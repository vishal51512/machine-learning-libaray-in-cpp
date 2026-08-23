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
The sample entry point is `nova-ml/main.cpp`. Replace `<path-to-eigen-headers>` with your Eigen include path (commonly `/usr/include/eigen3`).

```bash
g++ -std=c++17 \
  -I./nova-ml \
  -I./nova-ml/include \
  -I<path-to-eigen-headers> \
  ./nova-ml/main.cpp \
  -o nova-ml-demo

./nova-ml-demo
```

## C++ vs Python performance comparison

The table below compares this project (`RandomForestClassifier(10, 3)`) against Python scikit-learn (`RandomForestClassifier(n_estimators=10, max_depth=3)`) on the same generated dataset and train/test split used in `nova-ml/main.cpp`.

| Metric | Nova ML (C++) | Python (scikit-learn) |
| --- | ---: | ---: |
| Accuracy | 1.000 | 1.000 |
| Precision | 1.000 | 1.000 |
| Recall | 1.000 | 1.000 |
| F1-score | 1.000 | 1.000 |
| Specificity | 1.000 | 1.000 |
| Training time (ms, avg of 30 runs) | 4.805 | 10.956 |
| Inference time (ms, avg of 30 runs) | 0.012 | 0.926 |

Notes:
- Dataset: 200 synthetic 2D points, binary classes, 80/20 split, `random_state=42`.
- Classification quality is identical on this easy dataset; timing can vary by hardware, compiler flags, and Python environment.

## Project structure
- `nova-ml/`
  - `include/`
    - `base/` — Estimator interface
    - `core/` — Matrix/Vector wrappers
    - `linear_model/` — Linear & logistic models
    - `metrics/` — Accuracy and utilities
    - `model_selection/` — Train/test split
    - `neighbors/` — KNN classifier
    - `tree/` — Decision tree & random forest
    - `visualization/` — Plot helpers (optional)
  - `main.cpp` — Example usage
