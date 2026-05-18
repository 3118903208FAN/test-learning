# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   code
# FileName:      ai_1.py
# Author:        Gao
# Datetime:      2026/5/18 15:15
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def dm_LogisticRegression():
    # 1 获取数据
    data = pd.read_csv("breast-cancer-wisconsin.csv")
    # data.info()
    # 2 基本数据处理
    # 2.1 缺失值处理
    data = data.replace(to_replace="?", value=np.nan)
    data = data.dropna()

    # 2.2 确定特征值,目标值
    x = data.iloc[:, 1:-1]
    # print("‘x.head()-->\n’", x.head())
    y = data["Class"]
    # print("‘y.head()-->\n’", y.head())

    # 2.3 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

    # 3 特征工程(标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 机器学习(逻辑回归)
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 5 模型评估
    y_predict = estimator.predict(x_test)
    print("‘y_predict-->’", y_predict)
    accuracy = estimator.score(x_test, y_test)
    print("‘accuracy-->’", accuracy)

dm_LogisticRegression()
