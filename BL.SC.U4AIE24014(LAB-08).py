import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# A1
def summation_unit(inputs, weights, bias):
    return np.dot(inputs, weights) + bias


def step_activation(x):
    return 1 if x >= 0 else 0


def bipolar_step(x):
    return 1 if x >= 0 else -1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh_fn(x):
    return np.tanh(x)


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return x if x > 0 else 0.01 * x


def comparator(target, output):
    return target - output


# A2
def perceptron_train(X, y, weights, bias, lr, activation):

    errors = []

    for epoch in range(1000):

        total_error = 0

        for i in range(len(X)):

            net = summation_unit(X[i], weights, bias)

            out = activation(net)

            error = comparator(y[i], out)

            weights += lr * error * X[i]

            bias += lr * error

            total_error += error**2

        errors.append(total_error)

        if total_error <= 0.002:
            return weights, bias, epoch+1, errors

    return weights, bias, 1000, errors


# A3
def compare_activation_epochs(X, y):

    acts = {
        "Bipolar": bipolar_step,
        "Sigmoid": sigmoid,
        "ReLU": relu
    }

    results = {}

    for name, act in acts.items():

        w = np.array([0.2, -0.75], dtype=float)

        b = 10

        _, _, epochs, _ = perceptron_train(X, y, w, b, 0.05, act)

        results[name] = epochs

    return results


# A4
def learning_rate_analysis(X, y):

    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    epochs_list = []

    for lr in rates:

        w = np.array([0.2,-0.75], dtype=float)

        b = 10

        _, _, ep, _ = perceptron_train(X, y, w, b, lr, step_activation)

        epochs_list.append(ep)

    return rates, epochs_list


# A5
def customer_perceptron():

    X = np.array([
        [20,6,2,386],
        [16,3,6,289],
        [27,6,2,393],
        [19,1,2,110],
        [24,4,2,280],
        [22,1,5,167],
        [15,4,2,271],
        [18,4,2,274],
        [21,1,4,148],
        [16,2,4,198]
    ])

    y = np.array([1,1,1,0,1,0,1,1,0,0])

    w = np.random.rand(4)

    b = np.random.rand()

    return perceptron_train(X, y, w, b, 0.01, sigmoid)


# A6
def pseudo_inverse_method(X, y):

    X_bias = np.c_[np.ones(X.shape[0]), X]

    weights = np.linalg.pinv(X_bias).dot(y)

    return weights


# A7
def simple_backprop_AND():

    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])

    y = np.array([[0],[0],[0],[1]])

    np.random.seed(42)

    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)

    lr = 0.05

    for epoch in range(1000):

        hidden = sigmoid(np.dot(X,W1))
        output = sigmoid(np.dot(hidden,W2))

        error = y-output

        if np.sum(error**2) <= 0.002:
            return epoch+1, W1, W2

        d_output = error * output*(1-output)

        d_hidden = d_output.dot(W2.T)*hidden*(1-hidden)

        W2 += hidden.T.dot(d_output)*lr
        W1 += X.T.dot(d_hidden)*lr

    return 1000, W1, W2


# A8
def two_output_mapping(y):

    mapped = []

    for val in y:

        if val == 0:
            mapped.append([1,0])
        else:
            mapped.append([0,1])

    return np.array(mapped)


# A9
def mlp_logic(X, y):

    clf = MLPClassifier(hidden_layer_sizes=(2,),
                        activation='logistic',
                        max_iter=1000)

    clf.fit(X,y)

    preds = clf.predict(X)

    return preds


# A10
def mlp_project_dataset(X, y):

    clf = MLPClassifier(hidden_layer_sizes=(10,),
                        activation='relu',
                        max_iter=1000)

    clf.fit(X,y)

    return clf.score(X,y)


X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])


w = np.array([0.2,-0.75], dtype=float)
b = 10

fw, fb, ep, err = perceptron_train(X_and, y_and, w, b, 0.05, step_activation)

print("A2 Epochs:", ep)

plt.plot(err)
plt.show()


print("A3 Activation Comparison:", compare_activation_epochs(X_and,y_and))


rates, eps = learning_rate_analysis(X_and,y_and)

plt.plot(rates,eps)
plt.show()


fw, fb, ep, err = perceptron_train(X_xor,y_xor,w,b,0.05,step_activation)

print("A5 XOR Epochs:", ep)


print("A6 Customer Perceptron:", customer_perceptron()[2])


print("A7 Pseudo Inverse:",
      pseudo_inverse_method(X_and,y_and))


print("A8 Backprop AND:",
      simple_backprop_AND()[0])


print("A9 XOR Backprop:",
      simple_backprop_AND()[0])


print("A10 Two Output Mapping:",
      two_output_mapping(y_and))


print("A11 MLP AND:",
      mlp_logic(X_and,y_and))

print("A11 MLP XOR:",
      mlp_logic(X_xor,y_xor))



import pandas as pd

df = pd.read_csv(r"C:\Users\G Srinivas yadav\Desktop\SEM-4\ML\dataset.csv")

df = df.fillna(df.mean())

X_proj = df.drop(columns=["LABEL"]).values
y_proj = df["LABEL"].astype(int).values

print("A12 Project Dataset Accuracy:",
      mlp_project_dataset(X_proj,y_proj))
