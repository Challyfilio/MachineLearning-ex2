import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
# print(data.head())

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.show()


def get_X(df):
    ones = pd.DataFrame({'ones': np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)
    return data.iloc[:, :-1].iloc[:, :].values


def get_y(df):
    return np.array(df.iloc[:, -1])


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


X = get_X(data)
print(X.shape)
y = get_y(data)
print(y.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim(-0.1, 1.1)

plt.show()

theta = np.zeros(3)
print(theta)


def cost(theta, X, y):
    a = -y * np.log(sigmoid(X.dot(theta)))
    b = (1 - y) * np.log(1 - sigmoid(X.dot(theta)))
    return np.mean(a - b)


print(cost(theta, X, y))


def gradientDescent(theta, X, y):
    a = (X.T).dot((sigmoid(X.dot(theta)) - y))
    return (1 / len(X)) * a

print(gradientDescent(theta,X,y))
