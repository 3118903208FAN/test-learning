# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   demo
# FileName:      欠拟合与过拟合的正则化.py
# Author:       FanCuiChuan
# Datetime:     2026/5/18 18:41
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#Lasso回归,L1正则化
# def dm04():
#     # 1 准备数据x y(增加上噪声)
#     np.random.seed(666)
#     x = np.random.uniform(-3, 3, size=100)
#     y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
#     # 2 实例化L1正则化模型 做实验:alpha惩罚力度越来越大k值越来越小,返回会欠拟合
#     estimator = Lasso(alpha=0.005)
#     # 3 训练模型
#     X = x.reshape(-1, 1)
#     X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 数据增加二次项
#     estimator.fit(X3, y)
#     print('estimator.coef_', estimator.coef_)
#     # 4 模型预测
#     y_predict = estimator.predict(X3)
#     # 5 计算均方误差
#     myret = mean_squared_error(y, y_predict)
#     print('myret-->', myret)
#     # 6 画图
#     plt.scatter(x, y)
#     # 画图时输入的x数据: 要求是从小到大
#     plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
#     plt.show()
# dm04()

#Ridge回归，L2正则化

from sklearn.linear_model import Ridge
def dm05():
    # 1 准备数据x y(增加上噪声)
    np.random.seed(666)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2 实例化L2正则化模型
    estimator = Ridge(alpha=0.005)
    #3 训练模型
    X = x.reshape(-1, 1)
    X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 8, X ** 9, X ** 10])  # 数据增加二次项
    estimator.fit(X3, y)
    print('estimator.coef_', estimator.coef_)
    # 4 模型预测
    y_predict = estimator.predict(X3)
    # 5 计算均方误差
    myret = mean_squared_error(y, y_predict)
    print('myret-->', myret)
    # 6 画图
    plt.scatter(x, y)
    #画图时输入的x数据: 要求是从小到大
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
    plt.show()
dm05()