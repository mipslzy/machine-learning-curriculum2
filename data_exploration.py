# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# train_data_1 = pd.read_csv("train_public.csv")
# train_data_2 = pd.read_csv("train_internet.csv")
# test_data = pd.read_csv("test_public.csv")
#
# # 查看数据前三行信息
# print(train_data_1.head(3))
# print(train_data_2.head(3))
#
# # 查看数据集的整体信息
# print(train_data_1.info())
# print(train_data_2.info())
#
# # 描述性统计分析
# print(train_data_1.describe())
# print(train_data_2.describe())
#
# # 检查缺失值
# print(train_data_1.isnull())
# print(train_data_2.isnull())
#
# # 计算每个特征缺失值的数量
# missing_values1 = train_data_1.isnull().sum()
# missing_values2 = train_data_2.isnull().sum()
# print(missing_values1)
# print(missing_values2)
#
#
# # 获取train_data_1和train_data_2的字段集合
# fields_set_1 = set(train_data_1.columns)
# fields_set_2 = set(train_data_2.columns)
# #
# # 获取两个数据集的共同字段
# common_fields = fields_set_1.intersection(fields_set_2)
# #
# # 打印共同字段
# print(common_fields)
#
# # 数据可视化
# # 绘制直方图
# plt.hist(train_data_1)
# # 绘制散点图
# plt.scatter(train_data_1)
