import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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
    print("Feature coefficients: X1 =", theta1, ", X2 =", theta2)  # theta1, theta2
    print("Intercept:", intercept)        # theta0
    print("Accuracy of model on full dataset:", accuracy)
    
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
    print("Splitting data into 70:30 for training and testing")
    # split the data into training and testing data
    # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    model = train_log_regr(X, y)
    make_plot_with_predictions(X, y, model)


main()