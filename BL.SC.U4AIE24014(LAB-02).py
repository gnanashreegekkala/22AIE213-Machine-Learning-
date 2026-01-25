import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

#A1
def load_purchase_data(file):
    df = pd.read_excel(file, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values
    return X, y, df

def feature_matrix_rank(X):
    return np.linalg.matrix_rank(X)

def compute_cost_pseudoinverse(X, y):
    X_pinv = np.linalg.pinv(X)
    cost = X_pinv @ y
    return cost

#A2
def generate_labels(y):
    labels = []
    for val in y:
        if val > 200:
            labels.append(1)   
        else:
            labels.append(0)   
    return np.array(labels)

def train_linear_classifier(X, labels):
    X_aug = np.hstack((X, np.ones((X.shape[0], 1))))  # bias
    weights = np.linalg.pinv(X_aug) @ labels
    return weights

def predict_linear_classifier(X, weights):
    X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
    scores = X_aug @ weights
    predictions = []
    for s in scores:
        if s >= 0.5:
            predictions.append("RICH")
        else:
            predictions.append("POOR")
    return predictions

#A3
def mean_numpy(data):
    return np.mean(data)

def variance_numpy(data):
    return np.var(data)

def mean_manual(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)

def variance_manual(data):
    m = mean_manual(data)
    s = 0
    for x in data:
        s += (x - m) ** 2
    return s / len(data)

def average_execution_time(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / runs

#A4
def explore_dataset(df):
    missing = df.isnull().sum()
    return missing

#A5
def jaccard_and_smc(v1, v2):
    f11 = f00 = f10 = f01 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 0 and v2[i] == 0:
            f00 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1

    jc = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)
    return jc, smc

#A6
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#A7
def similarity_matrices(data):
    n = len(data)
    jc = np.zeros((n, n))
    smc = np.zeros((n, n))
    cos = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc[i][j], smc[i][j] = jaccard_and_smc(data[i], data[j])
            cos[i][j] = cosine_similarity(data[i], data[j])

    return jc, smc, cos

#A8
def impute_missing_values(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            if abs(df[col].skew()) < 1:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

#A9
def min_max_normalization(df):
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

#MAIN
def main():
    file = "Lab Session Data.xlsx"

    X, y, purchase_df = load_purchase_data(file)
    print("Rank of Feature Matrix:", feature_matrix_rank(X))
    print("Cost of Products using Pseudo-Inverse:", compute_cost_pseudoinverse(X, y))

    labels = generate_labels(y)
    weights = train_linear_classifier(X, labels)
    predictions = predict_linear_classifier(X, weights)
    print("Classifier Weights:", weights)
    print("Predicted Classes (first 10):", predictions[:10])

    stock_df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
    price = stock_df.iloc[:, 3].values

    print("Mean (NumPy):", mean_numpy(price))
    print("Variance (NumPy):", variance_numpy(price))
    print("Mean (Manual):", mean_manual(price))
    print("Variance (Manual):", variance_manual(price))

    print("Avg Time NumPy Mean:", average_execution_time(mean_numpy, price))
    print("Avg Time Manual Mean:", average_execution_time(mean_manual, price))

    wed_price = stock_df[stock_df["Day"] == "Wednesday"].iloc[:, 3]
    print("Wednesday Sample Mean:", mean_numpy(wed_price))

    apr_price = stock_df[stock_df["Month"] == "Apr"].iloc[:, 3]
    print("April Sample Mean:", mean_numpy(apr_price))

    chg = stock_df.iloc[:, 8]
    loss_prob = len(list(filter(lambda x: x < 0, chg))) / len(chg)
    print("Probability of Loss:", loss_prob)

    profit_wed = stock_df[(chg > 0) & (stock_df["Day"] == "Wednesday")]
    print("Probability of Profit on Wednesday:", len(profit_wed) / len(stock_df))

    plt.scatter(stock_df["Day"], chg)
    plt.title("Change % vs Day of Week")
    plt.show()

    thyroid = pd.read_excel(file, sheet_name="thyroid0387_UCI")
    summary, missing = explore_dataset(thyroid)
    print(summary)
    print("Missing Values:\n", missing)

    v1 = np.array([1, 0, 1, 1, 0])
    v2 = np.array([1, 1, 0, 1, 0])
    jc, smc = jaccard_and_smc(v1, v2)
    print("Jaccard Coefficient:", jc)
    print("Simple Matching Coefficient:", smc)
    print("Cosine Similarity:", cosine_similarity(v1, v2))

    data20 = thyroid.iloc[:20, :].fillna(0).values
    jc_m, smc_m, cos_m = similarity_matrices(data20)

    sns.heatmap(jc_m)
    plt.title("Jaccard Similarity Heatmap")
    plt.show()

    thyroid = impute_missing_values(thyroid)
    print("Missing after Imputation:\n", thyroid.isnull().sum())

    thyroid = min_max_normalization(thyroid)
    print("Data Normalization Completed")

if __name__ == "__main__":
    main()
