import numpy as np
import pandas as pd


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
    b = (1 - y) * np.log(1 - sigmoid(np.dot(x, theta)))  # (1-y)log(1-h(θ))
    return np.mean(a - b)


# 梯度下降
def gradientDescent(theta, x, y):
    a = np.dot(x.T, (sigmoid(np.dot(x, theta)) - y))  # (hθ(x)-y)x
    return (1 / len(x)) * a


def predict(x, theta):
    prob = sigmoid(np.dot(x, theta))
    return (prob >= 0.5).astype(int)


# 特征映射
def feature_mapping(x, y, power, as_ndarray=False):
    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
            for i in np.arange(power + 1)
            for p in np.arange(i + 1)
            }
    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)


# 正则化代价
def Regularized_Cost(theta, x, y, λ):
    theta_j1_to_n = theta[1:]
    regularized_term = (λ / (2 * len(x))) * np.power(theta_j1_to_n, 2).sum()
    return cost(theta, x, y) + regularized_term


# 正则化梯度
def Regularized_Gradient(theta, x, y, λ):
    theta_j1_to_n = theta[1:]  # 不加theta0
    regularized_theta = (λ / len(x)) * theta_j1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradientDescent(theta, x, y) + regularized_term
