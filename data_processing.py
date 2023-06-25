import pandas as pd

# 读取数据
train_data_1 = pd.read_csv("train_public.csv")
train_data_2 = pd.read_csv("train_internet.csv")
test_data = pd.read_csv("test_public.csv")
# 一、数据清洗
# 1.缺失值处理（用）
# 方案一：删除包含缺失值的行
# train_data_1.dropna(axis=0)
# train_data_2.dropna(axis=0)
# test_data.dropna(axis=0)

# 方案二：使用平均值填充数值型特征的缺失值
# train_data_1.fillna(train_data_1.mean())
# train_data_2.fillna(train_data_2.mean())

# 方案三：使用众数填充分类特征的缺失值
# train_data_1.fillna(train_data_1.mode().iloc[0])
# train_data_2.fillna(train_data_2.mode().iloc[0])

# 2.重复值处理（不用）
# 检测重复值
# print(train_data_1.duplicated())
# print(train_data_2.duplicated())
# 检测本训练集的数据不存在缺失值的情况
# 删除重复值
# train_data_1.drop_duplicates()
# train_data_2.drop_duplicates()

# 保留最后一个出现的重复行
# train_data_1.drop_duplicates(keep='last')
# train_data_2.drop_duplicates(keep='last')

# 3.异常值处理（暂时不用）
# 检测异常值（例如，使用标准差方法）
# mean = df['column_name'].mean()
# std = df['column_name'].std()
# threshold = mean + 3 * std
# outliers = df[df['column_name'] > threshold]
# 删除包含异常值的行
# df = df[df['column_name'] <= threshold]

# 二、数据类型转换
# 1.转换日期字段（需要）
# train_data_1['issue_date'] = pd.to_datetime(train_data_1['issue_date'])
# train_data_2['issue_date'] = pd.to_datetime(train_data_2['issue_date'])

# 2.转换数值字段（暂时不需要）
# data['column_name'] = data['column_name'].astype(float)

# 三、特征编码
# 独热编码
# encoded_data = pd.get_dummies(data, columns=['class', 'employer_type', 'industry'])
# 标签编码
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# data['class_encoded'] = label_encoder.fit_transform(data['class'])

# 四、特征缩放：
# 标准化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data['column_name'] = scaler.fit_transform(data['column_name'])
# 归一化
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data['column_name'] = scaler.fit_transform(data['column_name'])

# 五、特征选择
# 使用相关性进行特征选择
# correlation_matrix = data.corr()
# relevant_features = correlation_matrix[correlation_matrix['target'] > threshold]['feature_name']
# 使用特征重要性进行特征选择（以随机森林为例）
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
# importance = model.feature_importances_
# relevant_features = data.columns[importance > threshold]

# 六、数据集划分
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

