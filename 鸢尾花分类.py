# -*-coding:utf-8 -*- #
# ---------------------------------------------------------------------------
# ProjectName:   code
# FileName:      鸢尾花分类.py
# Author:        Gao
# Datetime:      2026/5/14 14:49
# Description:
# 命名规范：文件名全小写+下划线，类名大驼峰，方法和变量小写+下划线连接，
# 常量大写，变量和常量用名词，方法用动词
# ---------------------------------------------------------------------------
from sklearn.datasets import load_iris

# 1 查看数据情况
# def dm01_loadiris():
#     # 加载数据集
#     mydataset = load_iris()
#     # 查看数据集信息
#     print('\n查看数据集信息-->\n', mydataset.data[:6])
#     # 查看目标值
#     print('mydataset.target-->\n', mydataset.target)
#     #查看数据集的长度
#     print('mydataset.target的长度-->\n', len(mydataset.target))
#     # 查看目标值名字
#     print('mydataset.target_names-->\n', mydataset.target_names)
#     # 查看特征名
#     print('mydataset.feature_names-->\n', mydataset.feature_names)
#     # 查看数据集描述
#     print('\nmydataset.DESCR-->\n', mydataset.DESCR)
#     # 数据文件路径
#     print('mydataset.filename-->\n', mydataset.filename)
#
# dm01_loadiris()



# 2.
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# # 显示鸢尾花数据
# def dm02_showiris():
#     # 1 载入鸢尾花数据集 并显示特征名称.feature_names
#     mydataset = load_iris()
#     print(type(mydataset))
#     print(mydataset.feature_names)
#     # 2 把数据转换成dataframe格式 设置data, columns属性 目标值名称
#     iris_d = pd.DataFrame(mydataset['data'], columns=mydataset.feature_names)
#     iris_d['Species'] = mydataset.target
#
#     print('\niris_d-->\n', iris_d)
#     col1 = 'sepal length (cm)'
#     col2 = 'petal width (cm)'
#     # 3 sns.lmplot()显示
#     sns.lmplot(x=col1, y=col2, data=iris_d, hue='Species', fit_reg=False)
#     plt.xlabel(col1)
#     plt.ylabel(col2)
#     plt.title('iris')
#     plt.show()
#
# dm02_showiris()



from sklearn.model_selection import train_test_split
# # 数据集划分
# def dm03_traintest_split():
# # 1 加载数据集
#     mydataset = load_iris()
# # 2 划分数据集 *****
#     X_train, X_test, y_train, y_test =  train_test_split(
#   mydataset.data,
#         mydataset.target,
#         test_size=0.2,
#         random_state=22)
#     #
#     print('数据总数量', len(mydataset.data))
#     print('训练集中的x-特征值', len(X_train))
#     print('测试集中的x-特征值', len(X_test))
#     print(y_train)
# dm03_traintest_split()




from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# #
# #
# def dm04_():
#     # 1 获取数据集
#     mydataset = load_iris()
#
#     # 2 数据基本处理()
#     x_train, x_test, y_train, y_test =  train_test_split(mydataset.data, mydataset.target, test_size=0.2, random_state=22)
#
#     # 3 数据集预处理-数据标准化
#     transfer = StandardScaler()
#     x_train = transfer.fit_transform(x_train)
#     # 让测试集的均值和方法, 转换测试集数据;
#     x_test = transfer.transform(x_test)
#
#
#     # 4 机器学习(模型训练)
#
#     estimator = KNeighborsClassifier(n_neighbors=3)
#     estimator.fit(x_train, y_train)
#
#
#     # 5 模型评估 直接计算准确率 100个样本中模型预测对了多少
#     myscore = estimator.score(x_test, y_test)
#     print("myscore-->", myscore)
#
#
#     # 6 模型预测  # 需要对待预测数据,执行标准化
#     print('通过模型查看分类类别-->', estimator.classes_)
#     mydata =  [[5.1, 3.5, 1.4, 0.2],[4.6, 3.1, 1.5, 0.2]]
#     mydata = transfer.transform(mydata)
#     print('mydata-->', mydata)
#
#     mypred = estimator.predict(mydata)
#     print('mypred-->\n', mypred)
#
#     # mypred = estimator.predict_proba(mydata)
#     # print('mypred-->\n', mypred)
#
# dm04_()
# exercise
# def demo():
#     # 1.获取数据集
#     dataset=load_iris()
#     #2.数据的基本处理---->划分数据集（训练集和测试集）
#     x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.3,random_state=0)
#     # print(x_train,x_test,y_train,y_test)
#     # 3.特征值预处理---->标准化处理
#     transfer=StandardScaler()
#     x_train=transfer.fit_transform(x_train)
#     x_test=transfer.transform(x_test)
#     #4.模型的训练---->分类问题
#     model_class=KNeighborsClassifier(n_neighbors=3)
#     model_class.fit(x_train,y_train)
#
#     #5.模型的评估
#     score1=model_class.score(x_test,y_test)
#     print(score1)
#     #6.模型的预测
#     y_out = model_class.predict(x_test)
#     print(y_out,len(y_out))
#
# demo()



from sklearn.metrics import accuracy_score
# 1 获取数据集
mydataset = load_iris()
# 2 数据基本处理
x_train, x_test, y_train, y_test = train_test_split(mydataset.data, mydataset.target, test_size=0.2, random_state=22)
# 3 数据集预处理-数据标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
# 让测试集的均值和方法, 转换测试集数据;
x_test = transfer.transform(x_test)

# 4 机器学习(模型训练)
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train, y_train)

# 5-1 直接使用score函数 模型评估 100个样本中模型预测对了多少
myscore = estimator.score(x_test, y_test)
print('myscore-->', myscore)

# 5-2 利用sklearn.metrics包中的 accuracy_score 方法
y_predict = estimator.predict(x_test)
myresult = accuracy_score(y_test, y_predict)
print('myresult-->', myresult)


