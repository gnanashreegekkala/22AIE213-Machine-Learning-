import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lime.lime_tabular import LimeTabularExplainer

def load_data(path):

    df = pd.read_csv(path)

    df = df.fillna(df.mean())

    X = df.drop(columns=["LABEL"]).values
    y = df["LABEL"].astype(int).values

    return X, y

# A1
def stacking_model():

    base_models = [
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('nb', GaussianNB())
    ]

    meta_model = MLPClassifier(max_iter=1000)

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return stack

# A2
def create_pipeline(model):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    return pipe

# A3
def lime_explanation(model, X_train, X_test):

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=[str(i) for i in range(X_train.shape[1])],
        class_names=['0','1'],
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    return exp

# MAIN

X, y = load_data(r"C:\Users\G Srinivas yadav\Desktop\SEM-4\ML\dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

stack_model = stacking_model()
pipeline = create_pipeline(stack_model)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print("Pipeline Accuracy:", accuracy)
exp = lime_explanation(pipeline, X_train, X_test)
print("\nLIME Explanation:")
print(exp.as_list())