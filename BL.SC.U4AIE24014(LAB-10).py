import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lime.lime_tabular import LimeTabularExplainer
import shap
import warnings
warnings.filterwarnings("ignore")

# A1
def load_data(path):

    df = pd.read_csv(path)

    df = df.fillna(df.mean())

    X = df.drop(columns=["LABEL"])
    y = df["LABEL"].astype(int)

    return df, X, y


# A1
def correlation_heatmap(df):

    corr = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr)
    plt.title("Correlation Heatmap")
    plt.show()


# A2
def pca_99(X):

    pca = PCA(n_components=0.99)

    X_transformed = pca.fit_transform(X)

    return X_transformed


# A3
def pca_95(X):

    pca = PCA(n_components=0.95)

    X_transformed = pca.fit_transform(X)

    return X_transformed


# Model
def run_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return model, acc, X_train, X_test, y_train, y_test


# A4 
def sequential_feature_selection(X, y):

    model = RandomForestClassifier()

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=10,
        direction='forward',
        cv=None  
    )

    sfs.fit(X, y)

    return sfs.get_support()


# A5
def lime_explain(model, X_train, X_test):

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

    return exp.as_list()


# A5
def shap_explain(model, X_train):

    explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_train[:10])

    return shap_values


# MAIN

df, X, y = load_data(r"C:\Users\G Srinivas yadav\Desktop\SEM-4\ML\dataset.csv")

# A1
correlation_heatmap(df)

# Original
model_orig, acc_orig, X_train, X_test, y_train, y_test = run_model(X, y)
print("Original Accuracy:", acc_orig)

# A2
X_pca99 = pca_99(X)
_, acc_pca99, _, _, _, _ = run_model(X_pca99, y)
print("PCA 99% Accuracy:", acc_pca99)

# A3
X_pca95 = pca_95(X)
_, acc_pca95, _, _, _, _ = run_model(X_pca95, y)
print("PCA 95% Accuracy:", acc_pca95)

# A4
selected = sequential_feature_selection(X, y)
X_sfs = X.loc[:, selected]

_, acc_sfs, X_train_sfs, X_test_sfs, _, _ = run_model(X_sfs, y)
print("SFS Accuracy:", acc_sfs)

# A5 LIME
lime_output = lime_explain(model_orig, X_train, X_test)
print("LIME Explanation:", lime_output)

# A5 SHAP
shap_output = shap_explain(model_orig, X_train)
print("SHAP Explanation Generated")