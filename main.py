'''
建立分类器（求解θ0、θ1、θ2）
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from functions import *

# Logistic Regression
# data1
'''
data1 = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
# print(data1.head())
# sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data1,
#            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
# plt.show()

data1.insert(0, 'Ones', 1)

x = get_x(data1)
y = get_y(data1)

# print(X.shape)  # 100*3
# print(y.shape)  # 100*1
# print(X)
# print(y)

theta = np.zeros(3)
# sigmoid
# cost
print(cost(theta, x, y))
# gradientDescent
print(gradientDescent(theta, x, y))

# 拟合参数
res = opt.minimize(fun=cost, x0=theta, args=(x, y), method='Newton-CG', jac=gradientDescent)
print(res)

# 预测
final_theta = res.x
y_pred = predict(x, final_theta)
print(y_pred)

# 寻找决策边界
print(res.x)
coef = -(res.x / res.x[2])
print('coef', coef)

x = np.arange(150, step=0.1)
y = coef[0] + coef[1] * x

sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data1, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.plot(x, y, 'r')
plt.xlim(27, 102)
plt.ylim(27, 102)
plt.show()
'''

# Regularized logistic regression
# data2
λ = 1
power = 5

data2 = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
# sns.lmplot(x='test1', y='test2', hue='accepted', data=data,
#            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
# plt.show()

x1 = np.array(data2.test1)
x2 = np.array(data2.test2)
y = get_y(data2)


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


x = feature_mapping(x1, x2, power)
theta = np.zeros(x.shape[1])


# 正则化代价
def Regularized_cost(theta, x, y, λ=λ):
    theta_j1_to_n = theta[1:]
    regularized_term = (λ / (2 * len(x))) * np.power(theta_j1_to_n, 2).sum()
    return cost(theta, x, y) + regularized_term


# print(Regularized_cost(theta, x, y, λ=λ))

# 正则化梯度
def Regularized_gradient(theta, x, y, λ=λ):
    theta_j1_to_n = theta[1:]  # 不加theta0
    regularized_theta = (λ / len(x)) * theta_j1_to_n

    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradientDescent(theta, x, y) + regularized_term


# print(Regularized_gradient(theta, x, y, λ=λ))

res = opt.minimize(fun=Regularized_cost, x0=theta, args=(x, y), method='TNC', jac=Regularized_gradient)
final_theta = res.x
# print(res.x)


# 寻找决策边界
def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)

    mapped_cord = feature_mapping(x_cord, y_cord, power)
    inner_product = mapped_cord.values.dot(theta)
    decision = mapped_cord[np.abs(inner_product) < threshhold]
    print(decision)
    return decision.f10, decision.f01


def draw_boundary(power, final_theta):
    density = 1000
    threshhold = 2 * 10 ** -3

    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot(x='test1', y='test2', hue='accepted', data=df, height=6, fit_reg=False, scatter_kws={"s": 100})
    plt.scatter(x, y, c='r', s=10)
    plt.show()


draw_boundary(power, final_theta)
