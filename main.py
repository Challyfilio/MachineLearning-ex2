'''
建立分类器（求解θ0、θ1、θ2）
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from functions import *

λ = 0.01

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
# print(data.head())
# sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
# sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data,
#            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
# plt.show()

data.insert(0, 'Ones', 1)

x = get_x(data)
y = get_y(data)

# print(X.shape)  # 100*3
# print(y.shape)  # 100*1
# print(X)
# print(y)

theta = np.zeros(3)
# print(theta)

print(cost(theta, x, y))

print(gradientDescent(theta, x, y))
# 拟合参数
res = opt.minimize(fun=cost, x0=theta, args=(x, y), method='Newton-CG', jac=gradientDescent)
print(res)

final_theta = res.x
y_pred = predict(x, final_theta)
print(res.x)

# 寻找决策边界
coef = -(res.x / res.x[2])
print('coef', coef)

x = np.arange(150, step=0.1)
y = coef[0] + coef[1] * x

sns.set(context='notebook')  # style='ticks'
sns.lmplot(x='exam1', y='exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.plot(x, y, 'r')
plt.xlim(27, 102)
plt.ylim(27, 102)
plt.show()

# data2
'''
data = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])


# sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
# sns.lmplot(x='test1', y='test2', hue='accepted', data=data,
#            height=6, fit_reg=False, scatter_kws={"s": 50})  # fit_reg:是否显示拟合曲线
# plt.show()

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


x1 = np.array(data.test1)
x2 = np.array(data.test2)

df = feature_mapping(x1, x2, power=6)
# print(data.shape)
# print(data.head())
theta = np.zeros(df.shape[1])
x = feature_mapping(x1, x2, power=6)
print(x.shape)
y = get_y(data)
print(y.shape)


# 正则化代价
def regularized_cost(theta, x, y, λ=λ):
    theta_j1_to_n = theta[1:]
    regularized_term = (λ / (2 * len(x))) * np.power(theta_j1_to_n, 2).sum()
    return cost(theta, x, y) + regularized_term


print(regularized_cost(theta, x, y, λ=λ))


# 正则化梯度
def regularized_gradient(theta, x, y, λ=λ):
    theta_j1_to_n = theta[1:]  # 不加theta0
    regularized_theta = (λ / len(x)) * theta_j1_to_n

    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradientDescent(theta, x, y) + regularized_term


print(regularized_gradient(theta, x, y, λ=λ))

print('init cost ={}'.format(regularized_cost(theta, x, y, λ)))
res = opt.minimize(fun=regularized_cost, x0=theta, args=(x, y), method='Newton-CG', jac=regularized_gradient)
print(res)


def draw_boundary(power, λ):
    density = 1000
    threshhold = 2 * 10 ** -3
    final_theta = feature_mapped_logistic_regression(power)
    x, y = find_decision_boundary(density, power, final_theta, threshhold)

    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue='accepted', data=df, height=6, fit_reg=False, scatter_kws={"s": 100})

    plt.scatter(x, y, c='r', s=10)
    plt.show()


def feature_mapped_logistic_regression(power):
    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = get_y(df)

    x = feature_mapping(x1, x2, power)
    theta = np.zeros(x.shape[1])

    res = opt.minimize(fun=regularized_cost, x0=theta, args=(x, y), method='TNC', jac=regularized_gradient)
    final_theta = res.x

    return final_theta


def find_decision_boundary(density, power, theta, threshhold):
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)

    inner_product = mapped_cord.values.dot(theta)
    decision = mapped_cord[np.abs(inner_product) < threshhold]

    return decision.f10, decision.f01


draw_boundary(power=5, λ=λ)
'''
