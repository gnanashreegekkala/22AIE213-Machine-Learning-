import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

#A1
def entropy(y):
    counts = Counter(y)
    total = len(y)
    ent = 0

    for count in counts.values():
        p = count / total
        ent -= p * np.log2(p)

    return ent

#A2
def gini_index(y):
    counts = Counter(y)
    total = len(y)

    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p**2

    return gini

#A4
def equal_width_binning(feature, bins=4):

    min_val = np.min(feature)
    max_val = np.max(feature)

    width = (max_val - min_val) / bins

    binned = []

    for value in feature:
        bin_index = int((value - min_val) / width)

        if bin_index == bins:
            bin_index -= 1

        binned.append(bin_index)

    return np.array(binned)

def information_gain(X, y, feature_index):

    parent_entropy = entropy(y)

    feature = X[:, feature_index]
    values = np.unique(feature)

    weighted_entropy = 0

    for v in values:
        subset_y = y[feature == v]

        weight = len(subset_y) / len(y)
        weighted_entropy += weight * entropy(subset_y)

    ig = parent_entropy - weighted_entropy

    return ig

#A3
def find_root_feature(X, y):

    gains = []

    for i in range(X.shape[1]):
        ig = information_gain(X, y, i)
        gains.append(ig)

    root_index = np.argmax(gains)

    return root_index, gains

#A5
class SimpleDecisionTree:

    def __init__(self):
        self.tree = None

    def fit(self, X, y):

        root, gains = find_root_feature(X, y)

        self.tree = {
            "root_feature": root,
            "information_gain": gains[root]
        }

    def get_tree(self):
        return self.tree

#A6
def visualize_tree(X_train, y_train):

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    plt.figure(figsize=(12,6))
    plot_tree(model, filled=True)
    plt.show()

    return model

#A7
def plot_decision_boundary(X, y):

    # Use only first 2 features
    X = X[:, :2]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 500),
        np.arange(y_min, y_max, 500)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(X[:,0], X[:,1], c=y)

    plt.title("Decision Tree Decision Boundary")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

    plt.show()


#main
df = pd.read_csv("dataset.csv")
df = df.fillna(df.mean())
X = df.drop(columns=["LABEL"]).values
y = df["LABEL"].astype(int).values

X_binned = X.copy()

for i in range(X_binned.shape[1]):
    X_binned[:,i] = equal_width_binning(X_binned[:,i], bins=4)

dataset_entropy = entropy(y)
dataset_gini = gini_index(y)

print("Dataset Entropy:", dataset_entropy)
print("Dataset Gini Index:", dataset_gini)

root_feature, gains = find_root_feature(X_binned, y)

print("Root Feature Index:", root_feature)

tree = SimpleDecisionTree()
tree.fit(X_binned, y)

print("Tree Structure:", tree.get_tree())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = visualize_tree(X_train, y_train)
plot_decision_boundary(X, y)