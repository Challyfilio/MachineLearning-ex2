import numpy as np


def get_X(df):  # 读取特征
    return np.array(df.iloc[:, :-1])


def get_y(df):  # 读取标签
    return np.array(df.iloc[:, -1])


def normalize_feature(df):  # 特征缩放
    return df.apply(lambda column: (column - column.mean()) / column.std())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    a = -y * np.log(sigmoid(np.dot(X, theta)))  # -log(hθ(x))
    b = (1 - y) * np.log(1 - sigmoid(np.dot(X, theta)))  # -log(1-h(θ))
    return np.mean(a - b)
