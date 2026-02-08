import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)

#A1
def compute_classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)
    return cm, accuracy, precision, recall, f1


#A2
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


#A3
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, size=(n, 2))
    y = np.array([0 if x[0] + x[1] < 11 else 1 for x in X])
    return X, y


def plot_training_data(X, y):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1])
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Training Data Scatter Plot")
    plt.show()


#A4 & A5
def generate_test_grid(step=0.1):
    x_vals = np.arange(0, 10, step)
    y_vals = np.arange(0, 10, step)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_decision_boundary(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    xx, yy, grid = generate_test_grid()
    preds = model.predict(grid)

    plt.scatter(grid[preds == 0][:, 0], grid[preds == 0][:, 1], alpha=0.3)
    plt.scatter(grid[preds == 1][:, 0], grid[preds == 1][:, 1], alpha=0.3)

    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1])
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1])

    plt.title(f"Decision Boundary (k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


#A7
def tune_k_value(X_train, y_train):
    param_grid = {"n_neighbors": list(range(1, 21))}
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy"
    )
    grid.fit(X_train, y_train)
    return grid.best_params_, grid.best_score_


#main

df = pd.read_csv("dataset.csv")   # keep CSV in same folder
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

classes = np.unique(y)[:2]
mask = np.isin(y, classes)
X = X[mask]
y = y[mask]

# Convert labels to binary
label_map = {classes[0]: 0, classes[1]: 1}
y = np.array([label_map[val] for val in y])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

cm_train, acc_tr, prec_tr, rec_tr, f1_tr = compute_classification_metrics(
    y_train, train_preds
)
cm_test, acc_te, prec_te, rec_te, f1_te = compute_classification_metrics(
    y_test, test_preds
)

print("Training Confusion Matrix:\n", cm_train)
print("Training Accuracy:", acc_tr)
print("Training Precision:", prec_tr)
print("Training Recall:", rec_tr)
print("Training F1:", f1_tr)

print("\nTest Confusion Matrix:\n", cm_test)
print("Test Accuracy:", acc_te)
print("Test Precision:", prec_te)
print("Test Recall:", rec_te)
print("Test F1:", f1_te)

X_syn, y_syn = generate_training_data()
plot_training_data(X_syn, y_syn)

for k in [1, 3, 7]:
    plot_decision_boundary(X_syn, y_syn, k)

best_k, best_score = tune_k_value(X_train, y_train)
print("\nBest k from GridSearch:", best_k)
print("Best CV Accuracy:", best_score)
