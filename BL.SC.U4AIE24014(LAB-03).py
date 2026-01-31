import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#A1
def dot_product(A, B):
    return sum(a * b for a, b in zip(A, B))

def euclidean_norm(A):
    return (sum(a ** 2 for a in A)) ** 0.5

#A2
def mean_vector(X):
    return np.mean(X, axis=0)

def variance_vector(X):
    return np.var(X, axis=0)

def std_vector(X):
    return np.std(X, axis=0)

def interclass_distance(mean1, mean2):
    return np.linalg.norm(mean1 - mean2)

#A3
def plot_histogram(feature_data, bins=10):
    plt.hist(feature_data, bins=bins)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Feature")
    plt.show()

#A4
def feature_mean_variance(feature_data):
    return np.mean(feature_data), np.var(feature_data)

def minkowski_distance(A, B, p):
    return (sum(abs(a - b) ** p for a, b in zip(A, B))) ** (1 / p)

def minkowski_plot(A, B):
    p_values = range(1, 11)
    distances = [minkowski_distance(A, B, p) for p in p_values]
    plt.plot(p_values, distances, marker='o')
    plt.xlabel("p value")
    plt.ylabel("Distance")
    plt.title("Minkowski Distance vs p")
    plt.show()

#A10
def custom_knn_predict(X_train, y_train, test_point, k):
    distances = []
    for x, y in zip(X_train, y_train):
        dist = euclidean_norm(x - test_point)
        distances.append((dist, y))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [label for _, label in neighbors]
    return max(set(labels), key=labels.count)

def custom_knn_classifier(X_train, y_train, X_test, k):
    return np.array([custom_knn_predict(X_train, y_train, x, k) for x in X_test])

#A13
def confusion_matrix_manual(y_true, y_pred):
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0

#MAIN
df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

classes = np.unique(y)[:2]
mask = np.isin(y, classes)
X = X[mask]
y = y[mask]

label_map = {classes[0]: 0, classes[1]: 1}
y = np.array([label_map[val] for val in y])

#A1
A, B = X[0], X[1]
print("Dot Product (manual):", dot_product(A, B))
print("Dot Product (numpy):", np.dot(A, B))
print("Norm (manual):", euclidean_norm(A))
print("Norm (numpy):", np.linalg.norm(A))

#A2
X_class1 = X[y == classes[0]]
X_class2 = X[y == classes[1]]

mean1, mean2 = mean_vector(X_class1), mean_vector(X_class2)
std1, std2 = std_vector(X_class1), std_vector(X_class2)

print("Interclass Distance:", interclass_distance(mean1, mean2))

#A3
plot_histogram(X[:, 0])
print("Mean & Variance:", feature_mean_variance(X[:, 0]))

#A4
minkowski_plot(X[0], X[1])

#A5
print("Manual Minkowski:", minkowski_distance(X[0], X[1], 3))
print("Scipy Minkowski:", minkowski(X[0], X[1], 3))

#A6
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#A7
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#A8
print("sklearn kNN Accuracy:", knn.score(X_test, y_test))

#A9
print("Predictions:", knn.predict(X_test[:5]))

#A10
y_pred_custom = custom_knn_classifier(X_train, y_train, X_test, 3)

#A11
k_values = range(1, 12)
accuracies = []
for k in k_values:
    preds = custom_knn_classifier(X_train, y_train, X_test, k)
    acc = np.mean(preds == y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.show()

#A12
cm = confusion_matrix(y_test, y_pred_custom)
print("Confusion Matrix:\n", cm)

#A13
tp, tn, fp, fn = confusion_matrix_manual(y_test, y_pred_custom)
prec = precision(tp, fp)
rec = recall(tp, fn)
f1 = f1_score(prec, rec)

print("Accuracy:", accuracy(tp, tn, fp, fn))
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)