import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def make_plot_1A(X, y):    
    plt.figure(figsize=(7,6))  # larger figure  
    # Plot +1 points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='green', alpha=0.7, label='+1')

    # Plot -1 points
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='blue', alpha=0.7, label='-1')

    # Labels, title, legend
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("1A")
    plt.legend()
    plt.show()


def train_log_regr(X, y):  
    # Create and train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # relevent parameters
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    theta_str = ", ".join(f"X{i+1} = {np.round(theta, 5)}" for i, theta in enumerate(thetas))

    accuracy = model.score(X, y)

    # Print model parameters
    print(f"Feature coefficients: {theta_str}")  # theta1, theta2
    print(f"Intercept: {intercept:.5f}")        # theta0
    print(f"Accuracy of model on full dataset: {accuracy:.5f}")
    
    make_predictions(thetas)

    return model

def make_predictions(thetas):
     # Interpretation
    print("\nInterpretation:")
    for i, theta in enumerate(thetas):
        if theta > 0:
            print(f"- Feature X{i+1} increases the probability of predicting +1")
        elif theta < 0:
            print(f"- Feature X{i+1} decreases the probability of predicting +1")
        else:
            print(f"- Feature X{i+1} has no influence")
    
    # Most influential feature
    most_influential = np.argmax(np.abs(thetas))  # index of largest |theta|
    print(f"\nFeature X{most_influential+1} has the most influence on prediction")

def plot_log_regr_predictions(X, y, model):
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    theta1 = thetas[0]
    theta2 = thetas[1]

    plt.figure(figsize=(7,6))  # larger figure

    # Original data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='green', alpha=0.7, label='Actual +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='blue', alpha=0.7, label='Actual -1')

    # Predicted points
    y_pred = model.predict(X)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='+', color='red', alpha=0.5, linewidths=1.5, label='Predicted +1')
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='x', color='orange', alpha=0.8, linewidths=1, label='Predicted -1')

    # Decision boundary: X2 = -(theta0 + theta1*X1)/theta2
    x_min, x_max = X[:,0].min(), X[:,0].max()
    margin = 0.05 * (x_max - x_min)  # 5% of data range as margin
    x_values = np.linspace(x_min - margin, x_max + margin, 100)
    y_values = -(intercept + theta1*x_values)/theta2
    plt.plot(x_values, y_values, linestyle='--', color='purple', linewidth=2, label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Training Data and Logistic Regression Predictions")
    plt.legend()
    plt.show()

def train_linear_svm(X, y):
    C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    models = {}

    # Print header for the table
    print(f"{'C':>8} | {'weight_X1':>10} | {'weight_x2':>10} | {'bias':>10} | {'accuracy':>10}")
    print("-"*60)

    for C in C_values:
        model = LinearSVC(C=C, max_iter=10000)
        model.fit(X, y)

        # Extract model parameters
        weights = model.coef_[0]
        weight_X1 = weights[0]  # weight for X1
        weight_X2 = weights[1]  # weight for X2
        bias = model.intercept_[0]     # bias term
        acc = model.score(X, y)        # accuracy on training data

        # Print parameters in tabulated format
        print(f"{C:8.3f} | {weight_X1:10.5f} | {weight_X2:10.5f} | {bias:10.5f} | {acc:10.5f}")

        models[C] = model
    print('\n\n')

    return models

def plot_svm_predictions(X, y, svm_models):

    for C, model in svm_models.items():
        # relevent parameters
        weights = model.coef_[0]
        weight_X1 = weights[0]
        weight_x2 = weights[1]
        bias = model.intercept_[0]

        plt.figure(figsize=(7,6))

        # Original data points
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='green', alpha=0.7, label='Actual +1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='blue', alpha=0.7, label='Actual -1')

        # Predicted points
        y_pred = model.predict(X)
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='+', color='red', alpha=0.5, linewidths=1.5, label='Predicted +1')
        plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='x', color='orange', alpha=0.8, linewidths=1, label='Predicted -1')

        # Decision boundary: X2 = -(bias + weight_X1*X1)/weight_x2
        x_min, x_max = X[:,0].min(), X[:,0].max()
        margin = 0.05 * (x_max - x_min)  # 5% of data range as margin
        x_values = np.linspace(x_min - margin, x_max + margin, 100)
        y_values = -(bias + weight_X1 * x_values) / weight_x2
        plt.plot(x_values, y_values, linestyle='--', color='purple', linewidth=2, label='Decision boundary')

        plt.xlabel("X_1")
        plt.ylabel("X_2")
        plt.title(f"Linear SVM Predictions (C={C})")
        plt.legend()
        plt.show()

def train_log_regr_with_sq(X_sq, y):

    # Create and train model
    model = LogisticRegression()
    model.fit(X_sq, y)
    
    # relevent parameters
    intercept = model.intercept_[0]
    thetas = model.coef_[0]
    theta_str = ", ".join(f"X{i+1} = {np.round(theta, 5)}" for i, theta in enumerate(thetas))

    accuracy = model.score(X_sq, y)

    # Print model parameters
    print(f"Feature coefficients: {theta_str}")  # theta1, theta2, theta3, theta4
    print(f"Intercept: {intercept:.5f}")        # theta0
    print(f"Accuracy of model on full dataset: {accuracy:.5f}")

    make_predictions(thetas)

    return model
    
def plot_sq_predictions(X, y, X_sq, model):
    plt.figure(figsize=(7,6))  # larger figure

    # Original data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='green', alpha=0.7, label='Actual +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', color='blue', alpha=0.7, label='Actual -1')

    # Predicted points
    y_pred = model.predict(X_sq)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='+', color='red', alpha=0.5, linewidths=1.5, label='Predicted +1')
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='x', color='orange', alpha=0.8, linewidths=1, label='Predicted -1')

    # Decision boundary: X2 = -(theta0 + theta1*X1)/theta2
    x_min, x_max = X[:,0].min(), X[:,0].max()
    margin = 0.05 * (x_max - x_min)  # 5% of data range as margin
    #x_values = np.linspace(x_min - margin, x_max + margin, 100)
    #y_values = -(intercept + theta1*x_values)/theta2
    #plt.plot(x_values, y_values, linestyle='--', color='purple', linewidth=2, label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Logistic Regression Predictions with Added Squared Features")
    plt.legend()
    plt.show()

def baseline_accuracy(y):
    most_common = np.sign((y == 1).sum() - (y == -1).sum())
    baseline_pred = np.full_like(y, most_common)
    acc = np.mean(baseline_pred == y)
    print(f"Baseline (always predicts {most_common}): {acc:.5f}")
    return acc

def main():
    # path for csv file
    #csv_path = "C:\\Users\\Pri\\Documents\\GitHub\\CS7CS4\\Week2\\week2.csv"
    csv_path = "Week2/week2.csv"
    print("test")

    # Load CSV, skip comment line, no header
    df = pd.read_csv(csv_path, comment="#", header=None)
    print(df.head(), end="\n\n")  # preview data
    
    # Extract features and labels
    X1 = df.iloc[:,0] # feature 1
    X2 = df.iloc[:,1] # feature 2
    X = np.column_stack((X1, X2))  # first 2 columns as 2D array
    y = df.iloc[:, 2] # third column as labels

    make_plot_1A(X, y)

    # train Logistic Regression
    log_model = train_log_regr(X, y)
    plot_log_regr_predictions(X, y, log_model)

    # train SVM
    svm_models = train_linear_svm(X, y)
    plot_svm_predictions(X, y, svm_models)

    # train Logistic Regression with new features
    # Extend features for prediction
    X_sq = np.column_stack((X1, X2, X1**2, X2**2))
    sq_model = train_log_regr_with_sq(X_sq, y)
    plot_sq_predictions(X, y, X_sq, sq_model)

    baseline_accuracy(y)



main()