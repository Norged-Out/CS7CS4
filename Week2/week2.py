import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def make_plot_1A(X, y):
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
    
    b0 = model.intercept_
    b1 = model.coef_[0, 0]
    b2 = model.coef_[0, 1]

    # Print model parameters
    print("Feature coefficients: X1 =", b1, ", X2 =", b2)  # b1, b2
    print("Intercept:", b0)        # b0
    
    make_predictions(b1, b2)

    return model

def make_predictions(b1, b2):
     # Interpretation
    print("\nInterpretation:")
    if b1 > 0:
        print("- Feature X1 increases the probability of predicting +1")
    elif b1 < 0:
        print("- Feature X1 decreases the probability of predicting +1")
    else:
        print("- Feature X1 has no influence")

    if b2 > 0:
        print("- Feature X2 increases the probability of predicting +1")
    elif b2 < 0:
        print("- Feature X2 decreases the probability of predicting +1")
    else:
        print("- Feature X2 has no influence")
    
    # Most influential feature
    if abs(b1) > abs(b2):
        print("\nFeature X1 has the most influence on prediction")
    else:
        print("\nFeature X2 has the most influence on prediction")

def make_plot_with_predictions(X, y, model):
    b0 = model.intercept_[0]
    b1 = model.coef_[0, 0]
    b2 = model.coef_[0, 1]

    plt.figure(figsize=(7,6))  # larger figure

    # Original data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='green', label='Actual +1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', facecolors='none', edgecolors='blue', label='Actual -1')

    # Predicted points
    y_pred = model.predict(X)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], marker='x', color='red', label='Predicted +1')
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], marker='s', facecolors='none', edgecolors='orange', label='Predicted -1')

    # Decision boundary: X2 = -(b0 + b1*X1)/b2
    x_values = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 100)
    y_values = -(b0 + b1*x_values)/b2
    plt.plot(x_values, y_values, 'k--', label='Decision boundary')

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title("Training Data and Logistic Regression Predictions")
    plt.legend()
    plt.show()


def main():

    # path for csv file
    csv_path = "C:\\Users\\Pri\\Documents\\GitHub\\CS7CS4\\Week2\\week2.csv"

    # Load CSV, skip comment line, no header
    df = pd.read_csv(csv_path, comment="#", header=None)
    print(df.head(), end="\n\n")  # preview data
    
    # Extract features and labels
    X1 = df.iloc[:,0] # feature 1
    X2 = df.iloc[:,1] # feature 2
    X = np.column_stack((X1, X2))  # first 2 columns as 2D array
    y = df.iloc[:, 2] # third column as labels

    make_plot_1A(X, y)
    model = train_log_regr(X, y)
    make_plot_with_predictions(X, y, model)


main()