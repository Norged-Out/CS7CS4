'''
Author: Priyansh Nayak
CS7CS4 Machine Learning
Week 2 Assignment
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def plot_given_data(X, Y):
    '''
    Plot the given dataset with +1 and -1 points in different colors.
    '''  
    plt.figure(figsize=(7,6))  # larger figure  
    # Plot +1 points
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='cyan', alpha=0.7, label='+1')

    # Plot -1 points
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', color='orange', alpha=0.7, label='-1')

    # Labels, title, legend
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Dataset Visualization")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    #plt.savefig("Week2/given_data.png")


def train_log_regr(X, Y):
    '''
    Train a Logistic Regression model on the given data and print its parameters.
    '''
    # Create and train model
    model = LogisticRegression()
    model.fit(X, Y)

    # relevent parameters
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    # make a string of all the feature coefficients
    theta_str = ", ".join(f"X{i+1} = {np.round(theta, 5)}" for i, theta in enumerate(thetas))

    accuracy = model.score(X, Y) # no. of correct predictions / total predictions

    # print model parameters
    print("\nLogistic Regression Model Parameters:")
    print(f"Feature coefficients: {theta_str}")  # theta1, theta2
    print(f"Intercept: {intercept:.5f}")        # theta0
    print(f"Accuracy of model on full dataset: {accuracy:.5f}")
    
    make_predictions(thetas)

    return model


def make_predictions(thetas):
    '''
    Make predictions based on feature coefficients and print interpretations.
    '''
    # interpretation of feature coefficients for predicting +1
    print("\nInterpretation:")
    for i, theta in enumerate(thetas):
        if theta > 0:
            print(f"- Feature X{i+1} increases the probability of predicting +1")
        elif theta < 0:
            print(f"- Feature X{i+1} decreases the probability of predicting +1")
        else:
            print(f"- Feature X{i+1} has no influence")

    # most influential feature
    most_influential = np.argmax(np.abs(thetas))  # index of largest |theta|
    print(f"Feature X{most_influential+1} has the most influence on prediction\n")


def plot_log_regr_predictions(X, Y, model):
    '''
    Plot the logistic regression decision boundary and the data points.
    '''
    # relevent parameters
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    theta1 = thetas[0]
    theta2 = thetas[1]

    plt.figure(figsize=(8,6))  # larger figure

    # original data points
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='cyan', alpha=0.7, label='Actual +1')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', color='orange', alpha=0.7, label='Actual -1')

    # predicted points
    Y_pred = model.predict(X)
    plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], marker='+', color='red', alpha=0.6, linewidths=1.5, label='Predicted +1')
    plt.scatter(X[Y_pred == -1, 0], X[Y_pred == -1, 1], marker='x', color='green', alpha=0.7, linewidths=1, label='Predicted -1')

    # decision boundary: X2 = -(theta0 + theta1*X1)/theta2
    x_min, x_max = X[:,0].min(), X[:,0].max()
    margin = 0.05 * (x_max - x_min)  # 5% of data range as margin
    x_values = np.linspace(x_min - margin, x_max + margin, 100)
    y_values = -(intercept + theta1*x_values)/theta2
    plt.plot(x_values, y_values, linestyle='--', color='purple', linewidth=2, label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Training Data and Logistic Regression Predictions")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    #plt.savefig("Week2/logistic_regression_predictions.png")


def train_linear_svm(X, Y):
    '''
    Train a Linear SVM model on the given data and print its parameters.
    '''
    # try different C values
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    models = {}

    # print header for the table for visual comparison
    print("Linear SVM Model Parameters for different C values:")
    print(f"{'C':>8} | {'weight_X1':>10} | {'weight_x2':>10} | {'bias':>10} | {'accuracy':>10}")
    print("-"*60)

    for C in C_values:
        model = LinearSVC(C=C, max_iter=10000)
        model.fit(X, Y)

        # extract model parameters
        weights = model.coef_[0] 
        weight_X1 = weights[0]  # weight for X1
        weight_X2 = weights[1]  # weight for X2
        bias = model.intercept_[0]     # bias term
        acc = model.score(X, Y)        # accuracy on training data

        # print parameters in tabulated format for visual comparison
        print(f"{C:8.3f} | {weight_X1:10.5f} | {weight_X2:10.5f} | {bias:10.5f} | {acc:10.5f}")

        models[C] = model

    return models


def plot_svm_predictions(X, Y, svm_models):
    '''
    Plot the SVM decision boundaries and the data points.
    '''
    for C, model in svm_models.items():
        # relevent parameters
        weights = model.coef_[0]
        weight_X1 = weights[0]
        weight_x2 = weights[1]
        bias = model.intercept_[0]

        plt.figure(figsize=(8,6)) # larger figure

        # original data points
        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='cyan', alpha=0.7, label='Actual +1')
        plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', color='orange', alpha=0.7, label='Actual -1')

        # predicted points
        Y_pred = model.predict(X)
        plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], marker='+', color='red', alpha=0.6, linewidths=1.5, label='Predicted +1')
        plt.scatter(X[Y_pred == -1, 0], X[Y_pred == -1, 1], marker='x', color='green', alpha=0.7, linewidths=1, label='Predicted -1')

        # decision boundary: X2 = -(bias + weight_X1*X1)/weight_x2
        x_min, x_max = X[:,0].min(), X[:,0].max()
        margin = 0.05 * (x_max - x_min)  # 5% of data range as margin
        x_values = np.linspace(x_min - margin, x_max + margin, 100)
        y_values = -(bias + weight_X1 * x_values) / weight_x2
        plt.plot(x_values, y_values, linestyle='--', color='purple', linewidth=2, label='Decision boundary')

        plt.xlabel("X_1")
        plt.ylabel("X_2")
        plt.title(f"Linear SVM Predictions (C={C})")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
        #plt.savefig(f"Week2/svm_predictions_C_{C}.png")


def train_log_regr_with_sq(X_sq, Y):
    '''
    Train a Logistic Regression model with squared features on the given data and print its parameters.
    '''
    # create and train model with extended features
    model = LogisticRegression()
    model.fit(X_sq, Y)

    # relevent parameters
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    # make a string of all the feature coefficients
    theta_str = ", ".join(f"X{i+1} = {np.round(theta, 5)}" for i, theta in enumerate(thetas))

    accuracy = model.score(X_sq, Y) # no. of correct predictions / total predictions

    # print model parameters
    print("\nLogistic Regression with Squared Features Model Parameters:")
    print(f"Feature coefficients: {theta_str}")  # theta1, theta2, theta3, theta4
    print(f"Intercept: {intercept:.5f}")        # theta0
    print(f"Accuracy of model on full dataset: {accuracy:.5f}")

    make_predictions(thetas)

    return model


def plot_sq_predictions(X, Y, X_sq, model):
    '''
    Plot the logistic regression decision boundary and the data points.
    '''
    plt.figure(figsize=(8,6))  # larger figure

    # original data points
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker='o', color='cyan', alpha=0.7, label='Actual +1')
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker='o', color='orange', alpha=0.7, label='Actual -1')

    # predicted points
    Y_pred = model.predict(X_sq)
    plt.scatter(X[Y_pred == 1, 0], X[Y_pred == 1, 1], marker='+', color='red', alpha=0.6, linewidths=1.5, label='Predicted +1')
    plt.scatter(X[Y_pred == -1, 0], X[Y_pred == -1, 1], marker='x', color='green', alpha=0.7, linewidths=1, label='Predicted -1')

    # decision boundary: theta0 + theta1*X1 + theta2*X2 + theta3*X1^2 + theta4*X2^2 = 0
    # nonlinear boundary, so we need to plot a contour
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    # 5% of data range as margin
    x_margin = 0.05 * (x_max - x_min)
    x_min, x_max = x_min - x_margin, x_max + x_margin

    # create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # compute the decision on a grid
    X_sq_grid = np.column_stack((xx.ravel(), yy.ravel(), xx.ravel()**2, yy.ravel()**2))
    Z = model.decision_function(X_sq_grid) 
    Z = Z.reshape(xx.shape)

    # plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[0], colors='purple', linestyles='--', linewidths=2)

    # adding proxy line because contour doesn't create a legend entry
    plt.plot([], [], color='purple', linestyle='--', linewidth=2, label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Logistic Regression Predictions with Added Squared Features")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    #plt.savefig("Week2/logistic_regression_sq_predictions.png")


def baseline_accuracy(Y):
    '''
    Compute and print the baseline accuracy of a model that always predicts the most common class.
    '''
    most_common = np.sign((Y == 1).sum() - (Y == -1).sum()) # +1 if more +1s, -1 if more -1s
    baseline_pred = np.full_like(Y, most_common) # figure out which label is more common
    acc = np.mean(baseline_pred == Y) 
    print(f"Baseline (always predicts {most_common}): {acc:.5f}")
    return acc


def main():
    # path for csv file
    csv_path = "Week2/week2.csv"
    print("test")

    # eoad CSV, skip comment line, no header
    df = pd.read_csv(csv_path, comment="#", header=None)
    print(df.head(), end="\n\n")  # preview data
    
    # extract features and labels
    X1 = df.iloc[:,0] # feature 1
    X2 = df.iloc[:,1] # feature 2
    X = np.column_stack((X1, X2))  # first 2 columns as 2D array
    Y = df.iloc[:, 2] # third column as labels

    plot_given_data(X, Y)

    # train Logistic Regression
    log_model = train_log_regr(X, Y)
    plot_log_regr_predictions(X, Y, log_model)

    # train SVM
    svm_models = train_linear_svm(X, Y)
    plot_svm_predictions(X, Y, svm_models)

    # train Logistic Regression with new features
    # extend features for prediction
    X_sq = np.column_stack((X1, X2, X1**2, X2**2))
    sq_model = train_log_regr_with_sq(X_sq, Y)
    plot_sq_predictions(X, Y, X_sq, sq_model)

    baseline_accuracy(Y) # compare against trivial accuracy


main()