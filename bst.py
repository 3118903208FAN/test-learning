# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   code
# FileName:      bst.py
# Author:        Gao
# Datetime:      2026/5/18 08:49
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------

# from sklearn.datasets import load_boston                # 数据
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error          # 均方误差评估
from sklearn.linear_model import Ridge, RidgeCV

import pandas as pd
import numpy as np

from ML.鸢尾花分类 import transfer

# # 梯度下降方式
# def SGDRegressor_model2():
#     # 1.获取数据
#     # data = load_boston()
#     data = pd.read_csv('bst.csv')
#
#     # print(data)
#     # 2.数据集划分
#     x_train, x_test, y_train, y_test = train_test_split(data, data.target, random_state=22)
#
#
#     # 3.特征工程-标准化
#     transfer = StandardScaler()
#     x_train = transfer.fit_transform(x_train)
#     x_test = transfer.transform(x_test)
#
#     # 4.机器学习-线性回归(梯度下降)
#     # estimator = SGDRegressor()
#     estimator = SGDRegressor(max_iter=1000, learning_rate="constant", eta0=0.005)
#
#     estimator.fit(x_train, y_train)
#     # 5.模型评估 # 5.1 获取模型参数
#     y_predict = estimator.predict(x_test)
#
#     # print("预测值为:\n", y_predict)
#     print("模型的权重系数为:\n", estimator.coef_)
#     print("模型的偏置为:\n", estimator.intercept_)
#     # 5.2 评价 均方误差
#     error = mean_squared_error(y_test, y_predict)
#     print("误差为:\n", error)
# SGDRegressor_model2()
# print("*"*30)

# #练习1：梯度下降方式：
# #1.读数据
# data=pd.read_csv('bst.csv')
# #2.划分数据
# x_train,x_test,y_train,y_test=train_test_split(data,data["target"],random_state=12)
# #3.数据标准化处理
# transfer=StandardScaler()
# x_train=transfer.fit_transform(x_train)
# x_test=transfer.transform(x_test)
# #4.模型对象的实例化
# model1=SGDRegressor()
# model1.fit(x_train,y_train)
# print("权重:",model1.coef_, "偏置:",model1.intercept_)
# #5.模型评估
# error=mean_squared_error(y_test,model1.predict(x_test))
# print("误差为：",error)

# 正规方程

# def linear_model1():
#     # 1 获取数据
#     # data = load_boston()    # 2 数据集划分
#     data = pd.read_csv('bst.csv')
#
#     x_train, x_test, y_train, y_test = train_test_split(data, data.target, random_state=22)
#
#     # 3 特征工程-标准化
#     transfer = StandardScaler()
#     x_train = transfer.fit_transform(x_train)
#     x_test = transfer.transform(x_test)
#
#     # 4 机器学习-线性回归(正规方程)
#     estimator = LinearRegression()
#     estimator.fit(x_train, y_train)
#
#     # 5 模型评估 # 获取模型系数
#     y_predict = estimator.predict(x_test)
#
#     # print("“预测值为:\n”", y_predict)
#     print("模型的权重系数为:\n", estimator.coef_)
#     print("模型的偏置为:\n", estimator.intercept_)
#     # 5.2 评价 均方误差
#     error = mean_squared_error(y_test, y_predict)
#     print("误差为:\n", error)
#
# linear_model1()


# #练习2：正规方程求解：
# #1.读取数据：
# data=pd.read_csv('bst.csv')
# #2.数据集的划分：
# x_train,x_test,y_train,y_test=train_test_split(data,data.target,random_state=22)
# #3.特征工程,对特征值进行预处理,进行标准化处理
# SD1=StandardScaler()
# x_train=SD1.fit_transform(x_train)
# x_test=SD1.transform(x_test)
# #4.模型对象的实例化（正规方程）
# model2=LinearRegression()
# model2.fit(x_train,y_train)
# y_predict=model2.predict(x_test)
# y_predict=y_predict.reshape(-1,1)
# print("预测值为：",y_predict)
# #5.模型的评估：
# error=mean_squared_error(y_test,y_predict)
# print("误差:",error)


















