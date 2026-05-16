# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   demo
# FileName:      ml2.py
# Author:       FanCuiChuan
# Datetime:     2026/5/14 14:46
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
from copy import deepcopy

from sklearn.preprocessing import StandardScaler,MinMaxScaler
data=[[4,5],[6,7],[8,9]]
# 1.归一化预处理
# scaler = MinMaxScaler()
# data=scaler.fit_transform(data)
# print(data)

#2.标准化预处理
scaler = StandardScaler()
data=scaler.fit_transform(data)
print(data)
