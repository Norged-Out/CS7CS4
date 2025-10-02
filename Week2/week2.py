import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def make_plot_1A(X, y):    
    plt.figure(figsize=(7,6))  # larger figure  
    # Plot +1 points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='green', label='+1')

    # Plot -1 points
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', facecolors='none', edgecolors='blue', label='-1')

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
    
    intercept = model.intercept_
    theta1 = model.coef_[0, 0]
    theta2 = model.coef_[0, 1]

    accuracy = model.score(X, y)

    # Print model parameters
    print(f"Feature coefficients: X1 = {theta1}, X2 = {theta2}")  # theta1, theta2
    print(f"Intercept: {intercept}")        # theta0
    print(f"Accuracy of model on full dataset: {accuracy}")
    
    make_predictions(theta1, theta2)

    return model

def make_predictions(theta1, theta2):
     # Interpretation
    print("\nInterpretation:")
    if theta1 > 0:
        print("- Feature X1 increases the probability of predicting +1")
    elif theta1 < 0:
        print("- Feature X1 decreases the probability of predicting +1")
    else:
        print("- Feature X1 has no influence")

    if theta2 > 0:
        print("- Feature X2 increases the probability of predicting +1")
    elif theta2 < 0:
        print("- Feature X2 decreases the probability of predicting +1")
    else:
        print("- Feature X2 has no influence")
    
    # Most influential feature
    if abs(theta1) > abs(theta2):
        print("\nFeature X1 has the most influence on prediction")
    else:
        print("\nFeature X2 has the most influence on prediction")

def make_plot_with_predictions(X, y, model):
    intercept = model.intercept_[0]
    theta1 = model.coef_[0, 0]
    theta2 = model.coef_[0, 1]

    plt.figure(figsize=(7,6))  # larger figure

    # Original data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='green', label='Actual +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', facecolors='none', edgecolors='blue', label='Actual -1')

    # Predicted points
    y_pred = model.predict(X)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='s', s=10, alpha=0.6, facecolors='none', edgecolors='red', label='Predicted +1')
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='s', s=10, alpha=0.6, facecolors='none', edgecolors='orange', label='Predicted -1')

    # Decision boundary: X2 = -(theta0 + theta1*X1)/theta2
    x_values = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 100)
    y_values = -(intercept + theta1*x_values)/theta2
    plt.plot(x_values, y_values, 'k--', label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Training Data and Logistic Regression Predictions")
    plt.legend()
    plt.show()

def train_linear_svm(X, y):
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    models = {}

    # Print header for the table
    print(f"{'C':>8} | {'weight_X1':>10} | {'weight_x2':>10} | {'bias':>10} | {'accuracy':>10}")
    print("-"*60)

    for C in C_values:
        model = LinearSVC(C=C, max_iter=10000)
        model.fit(X, y)

        # Extract model parameters using your symbols
        weight_X1 = model.coef_[0, 0]  # weight for X1
        weight_x2 = model.coef_[0, 1]  # weight for X2
        bias = model.intercept_[0]     # bias term
        acc = model.score(X, y)        # accuracy on training data

        # Print parameters in tabulated format
        print(f"{C:8.3f} | {weight_X1:10.4f} | {weight_x2:10.4f} | {bias:10.4f} | {acc:10.3f}")

        models[C] = model

    return models

def plot_svm_predictions(X, y, svm_models):

    for C, model in svm_models.items():
        weight_X1 = model.coef_[0, 0]
        weight_x2 = model.coef_[0, 1]
        bias = model.intercept_[0]

        plt.figure(figsize=(7,6))

        # Original data points
        plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='green', label='Actual +1')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', facecolors='none', edgecolors='blue', label='Actual -1')

        # Predicted points
        y_pred = model.predict(X)
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='s', s=20, alpha=0.6, facecolors='none', edgecolors='red', label='Predicted +1')
        plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='s', s=20, alpha=0.6, facecolors='none', edgecolors='orange', label='Predicted -1')

        # Decision boundary: X2 = -(bias + weight_X1*X1)/weight_x2
        x_vals = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 100)
        y_vals = -(bias + weight_X1 * x_vals) / weight_x2
        plt.plot(x_vals, y_vals, 'k--', label='Decision boundary')

        plt.xlabel("X_1")
        plt.ylabel("X_2")
        plt.title(f"Linear SVM Predictions (C={C})")
        plt.legend()
        plt.show()



def main():
    # path for csv file
    #csv_path = "C:\\Users\\Pri\\Documents\\GitHub\\CS7CS4\\Week2\\week2.csv"
    csv_path = "Week2/week2.csv"

    # Load CSV, skip comment line, no header
    df = pd.read_csv(csv_path, comment="#", header=None)
    print(df.head(), end="\n\n")  # preview data
    
    # Extract features and labels
    X1 = df.iloc[:,0] # feature 1
    X2 = df.iloc[:,1] # feature 2
    X = np.column_stack((X1, X2))  # first 2 columns as 2D array
    y = df.iloc[:, 2] # third column as labels

    make_plot_1A(X, y)
    #print("Splitting data into 70:30 for training and testing")
    # split the data into training and testing data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    log_model = train_log_regr(X, y)
    make_plot_with_predictions(X, y, log_model)

    #train SVM
    svm_models = train_linear_svm(X, y)
    plot_svm_predictions(X, y, svm_models)


main()