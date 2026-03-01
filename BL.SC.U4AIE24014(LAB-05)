import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

#A2

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2


#A1

def linear_regression_single(X_train, X_test, y_train, y_test):

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    return train_metrics, test_metrics


#A3

def linear_regression_multi(X_train, X_test, y_train, y_test):

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    train_metrics = regression_metrics(y_train, y_train_pred)
    test_metrics = regression_metrics(y_test, y_test_pred)

    return train_metrics, test_metrics


#A4&A5

def kmeans_scores(X, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_

    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)

    return sil, ch, db


#A6

def clustering_score_plot(X):

    k_values = range(2, 11)

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_values:
        sil, ch, db = kmeans_scores(X, k)
        sil_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)

    plt.plot(k_values, sil_scores)
    plt.title("Silhouette Score vs k")
    plt.show()

    plt.plot(k_values, ch_scores)
    plt.title("CH Score vs k")
    plt.show()

    plt.plot(k_values, db_scores)
    plt.title("DB Index vs k")
    plt.show()


#A7

def elbow_plot(X):

    distortions = []

    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    plt.plot(range(2, 20), distortions)
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.show()


#MAIN

df = pd.read_csv("dataset.csv")
y = df["0"].values                     
X = df.drop(columns=["0", "LABEL"]).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# A1
train_m, test_m = linear_regression_single(
    X_train[:, [0]], X_test[:, [0]], y_train, y_test
)
print("Single Feature Train:", train_m)
print("Single Feature Test:", test_m)

# A3
train_m2, test_m2 = linear_regression_multi(
    X_train, X_test, y_train, y_test
)
print("Multi Feature Train:", train_m2)
print("Multi Feature Test:", test_m2)

# A6
clustering_score_plot(X)

# A7
elbow_plot(X)