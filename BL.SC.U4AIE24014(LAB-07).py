import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def load_dataset(file_path):

    df = pd.read_csv(file_path)

    # Handle missing values
    df = df.fillna(df.mean())

    X = df.drop(columns=["LABEL"]).values
    y = df["LABEL"].astype(int).values

    return X, y

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return train_acc, test_acc

#A1
def prepare_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

#A2
def tune_random_forest(X_train, y_train):

    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }

    rf = RandomForestClassifier()

    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

#A3
def compare_models(X_train, X_test, y_train, y_test):

    models = {
        "SVM": SVC(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=300),
        "AdaBoost": AdaBoostClassifier()
    }

    results = {}

    for name, model in models.items():

        train_acc, test_acc = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        results[name] = (train_acc, test_acc)

    return results


#main

X, y = load_dataset("dataset.csv")
X_train, X_test, y_train, y_test = prepare_data(X, y)
results = compare_models(X_train, X_test, y_train, y_test)

print("\nModel Comparison (Train vs Test Accuracy):\n")
for model, scores in results.items():
    print(model, "-> Train:", scores[0], "| Test:", scores[1])

best_model, best_params = tune_random_forest(X_train, y_train)

train_acc, test_acc = evaluate_model(
    best_model, X_train, X_test, y_train, y_test
)

print("\nBest RandomForest After Tuning:")
print("Parameters:", best_params)
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)