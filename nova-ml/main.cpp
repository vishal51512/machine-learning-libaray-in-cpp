#include <iostream>
#include <algorithm>
#include <random>
#include <cstdlib>

#include "include/core/matrix.hpp"
#include "include/core/vector.hpp"

#include "include/base/estimator.hpp"

#include "include/linear_model/linear_regression.hpp"
#include "include/linear_model/ridge_regression.hpp"
#include "include/linear_model/polynomial_regression.hpp"
#include "include/linear_model/logistic_regression.hpp"

#include "include/model_selection/train_test_split.hpp"
#include "include/metrics/metrics.hpp"
#include "include/metrics/accuracy.hpp"

#include "include/neighbors/knn_classifier.hpp"

#include "include/tree/decision_tree_classifier.hpp"
#include "include/tree/random_forest_classifier.hpp"

// #include "include/visualization/plot.hpp"



using namespace nova;

int main() {
   // setenv("MPLBACKEND", "TkAgg", 1);

    // ===== Create Dataset =====
Matrix X(200,2);
Vector y(200);

// Class 0 (circle center at 0.3,0.3)
for(int i=0;i<100;i++) {
    double x1 = (double)rand()/RAND_MAX * 0.5;
    double x2 = (double)rand()/RAND_MAX * 0.5;

    X(i,0) = x1;
    X(i,1) = x2;
    y(i) = 0;
}

// Class 1 (circle center at 0.7,0.7)
for(int i=100;i<200;i++) {
    double x1 = 0.5 + (double)rand()/RAND_MAX * 0.5;
    double x2 = 0.5 + (double)rand()/RAND_MAX * 0.5;

    X(i,0) = x1;
    X(i,1) = x2;
    y(i) = 1;
}



    // ===== Split =====
    auto split = train_test_split(X,y,0.2,true,42);

    // ===== Base Estimator Pointer =====
    Estimator* model;

    // choose model here
    // model = new LinearRegression();
    // model = new RidgeRegression(0.5);
    // model = new PolynomialRegression(3);
    // model = new LogisticRegression(0.1, 2000);
    // model = new KNNClassifier(3);
    // model = new DecisionTreeClassifier(3);
    model = new RandomForestClassifier(10,3);

    // ===== Train =====
    model->fit(split.X_train, split.y_train);

    // ===== Predict =====
    auto preds = model->predict(split.X_test);
   //  nova::viz::decision_boundary(*model,0.0,1.0);
     // ploting
     

    // ===== Evaluate =====
    double acc =
    metrics::accuracy(split.y_test, preds);

std::cout << "Accuracy: " << acc << std::endl;


    // ===== Cleanup =====
    delete model;

    return 0;
}
