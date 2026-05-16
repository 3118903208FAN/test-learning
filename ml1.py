# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   demo
# FileName:      ml1.py
# Author:       FanCuiChuan
# Datetime:     2026/5/14 11:06
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
# 1.导包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# 2.实例化类
model_class=KNeighborsRegressor(n_neighbors=2)
# 3.模型训练
# x=[[1],[5],[6],[7]]
# y=[0,0,1,1]
x=[[1],[5],[6],[7]]
y=[0.1,0.2,0.3,0.4]
model_class.fit(x,y)
# 4.模型测试,预测结果
print(model_class.predict([[2]]))

#xiugai--------------------git