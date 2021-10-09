import numpy as np


# 读取特征
def get_X(df):
    return np.array(df.iloc[:, :-1])


# 读取标签
def get_y(df):
    return np.array(df.iloc[:, -1])


# 特征缩放
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    a = -y * np.log(sigmoid(np.dot(X, theta)))  # -log(hθ(x))
    b = (1 - y) * np.log(1 - sigmoid(np.dot(X, theta)))  # -log(1-h(θ))
    return np.mean(a - b)


# 梯度下降
def gradientDescent(theta, X, y):
    a = np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
    return (1 / len(X)) * a
