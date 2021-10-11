import numpy as np


# 读取特征
def get_x(df):
    return np.array(df.iloc[:, :-1])


# 读取标签
def get_y(df):
    return np.array(df.iloc[:, -1])


# 特征缩放
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, x, y):
    a = -y * np.log(sigmoid(np.dot(x, theta)))  # -ylog(hθ(x))
    b = (1 - y) * np.log(1 - sigmoid(np.dot(x, theta)))  # -(1-y)log(1-h(θ))
    return np.mean(a - b)


# 梯度下降
def gradientDescent(theta, x, y):
    a = np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))
    return (1 / len(x)) * a


def predict(x, theta):
    prob = sigmoid(np.dot(x, theta))
    return (prob >= 0.5).astype(int)
