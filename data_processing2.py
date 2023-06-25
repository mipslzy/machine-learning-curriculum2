import warnings
import pandas as pd
import datetime
warnings.filterwarnings('ignore')

# 训练数据加载
train_bank = pd.read_csv('train_public.csv')
train_internet = pd.read_csv('train_internet.csv')

# 测试数据加载
test = pd.read_csv("test_public.csv")

# 一、数据预处理
# 获取两个数据集的共同字段
# common_cols = []
# for col in train_bank.columns:
#     if col in train_internet.columns:
#         common_cols.append(col)
#     else: continue
# print(common_cols)

# 获取train_bank和train_internet的字段集合
fields_set_1 = set(train_bank.columns)
fields_set_2 = set(train_internet.columns)
# 获取两个数据集的共同字段
common_cols = fields_set_1.intersection(fields_set_2)
# print(common_cols)
# 获取两个数据集的不同字段
# train_bank_left = list(set(list(train_bank.columns)) - common_cols)
# train_internet_left = list(set(list(train_internet.columns)) - common_cols)

# print(train_bank_left)
# print(train_internet_left)

# 筛选出包含共同列的数据集
train1_data = train_bank.filter(items=common_cols)
train2_data = train_internet.filter(items=common_cols)
test_data = test.filter(items=common_cols)

# 二、日期数据处理
# 1.转换成标准日期类型
train1_data['issue_date'] = pd.to_datetime(train1_data['issue_date'])
train2_data['issue_date'] = pd.to_datetime(train2_data['issue_date'])
test_data['issue_date'] = pd.to_datetime(test_data['issue_date'])

# 2.提取多尺度特征
train1_data['issue_date_y'] = train1_data['issue_date'].dt.year
train1_data['issue_date_m'] = train1_data['issue_date'].dt.month

train2_data['issue_date_y'] = train2_data['issue_date'].dt.year
train2_data['issue_date_m'] = train2_data['issue_date'].dt.month

test_data['issue_date_y'] = test_data['issue_date'].dt.year
test_data['issue_date_m'] = test_data['issue_date'].dt.month

# 3.提取时间diff，设置初始时间，转换为天为单位
base_time = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')

train1_data['issue_date_diff'] = train1_data['issue_date'].apply(lambda x: x-base_time).dt.days
train1_data.drop('issue_date', axis = 1, inplace = True)

train2_data['issue_date_diff'] = train2_data['issue_date'].apply(lambda x: x-base_time).dt.days
train2_data.drop('issue_date', axis = 1, inplace = True)

test_data['issue_date_diff'] = test_data['issue_date'].apply(lambda x: x-base_time).dt.days
test_data.drop('issue_date', axis = 1, inplace = True)

# 三、其他数据的处理
employer_type = train1_data['employer_type'].value_counts().index
industry = train1_data['industry'].value_counts().index

emp_type_dict = dict(zip(employer_type, [0,1,2,3,4,5]))
industry_dict = dict(zip(industry, [i for i in range(15)]))

train1_data['work_year'].fillna('10+ years', inplace=True)
train2_data['work_year'].fillna('10+ years', inplace=True)
test_data['work_year'].fillna('10+ years', inplace=True)

work_year_map = {'10+ years': 10, '2 years': 2, '< 1 year': 0, '3 years': 3, '1 year': 1,
     '5 years': 5, '4 years': 4, '6 years': 6, '8 years': 8, '7 years': 7, '9 years': 9}

train1_data['work_year']  = train1_data['work_year'].map(work_year_map)
train2_data['work_year']  = train2_data['work_year'].map(work_year_map)
test_data['work_year']  = test_data['work_year'].map(work_year_map)

train1_data['class'] = train1_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
train2_data['class'] = train2_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
test_data['class'] = test_data['class'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})

train1_data['employer_type'] = train1_data['employer_type'].map(emp_type_dict)
train2_data['employer_type'] = train2_data['employer_type'].map(emp_type_dict)
test_data['employer_type'] = test_data['employer_type'].map(emp_type_dict)

train1_data['industry'] = train1_data['industry'].map(industry_dict)
train2_data['industry'] = train2_data['industry'].map(industry_dict)
test_data['industry'] = test_data['industry'].map(industry_dict)

# 将筛选后的数据集存储到新的文件中
train1_data.to_csv('train1_data.csv', index=False)
train2_data.to_csv('train2_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)