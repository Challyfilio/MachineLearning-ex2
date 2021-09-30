'''
建立分类器（求解θ0、θ1、θ2）
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])


# sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
# sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data,
#            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
# plt.show()

data.insert(0, 'Ones', 1)
# print(data.head())

def get_X(df):  # 读取特征
    # ones = pd.DataFrame({'ones': np.ones(len(df))}) #插入1列1
    # data = pd.concat([ones, df], axis=1)
    return data.iloc[:, :-1].iloc[:, :].values


def get_y(df):  # 读取标签
    return np.array(df.iloc[:, -1])


def normalize_feature(df):  # 特征缩放
    return df.apply(lambda column: (column - column.mean()) / column.std())


X = get_X(data)
y = get_y(data)


# print(X.shape)  # 100*3
# print(y.shape)  # 100*1
# print(X)
# print(y)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


'''
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim(-0.1, 1.1)
plt.show()
'''
theta = np.zeros(3)
print(theta)


def cost(theta, X, y):
    a = -y * np.log(sigmoid(X.dot(theta)))
    b = (1 - y) * np.log(1 - sigmoid(X.dot(theta)))
    return np.mean(a - b)


print(cost(theta, X, y))

'''
def gradientDescent(theta, X, y):
    a = (X.T).dot((sigmoid(X.dot(theta)) - y))
    return (1 / len(X)) * a


print(gradientDescent(theta, X, y))

import scipy.optimize as opt
from sklearn.metrics import classification_report

res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradientDescent)
print(res)


def predict(x, theta):
    prob = sigmoid(x.dot(theta))
    return (prob >= 0.5).astype(int)


final_theta = res.x
y_pred = predict(X, final_theta)
print(classification_report(y, y_pred))

print(res.x)
coef = -(res.x / res.x[2])
print(coef)

x = np.arange(130, step=0.1)
y = coef[0] + coef[1] * x

sns.set(context='notebook')  # style='ticks'
sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.plot(x, y, 'r')
plt.xlim(27, 102)
plt.ylim(27, 102)
plt.show()
'''
