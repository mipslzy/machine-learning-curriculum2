from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 加载数据
train1_data = pd.read_csv('train1_data.csv')
train2_data = pd.read_csv('train2_data.csv')
test_data = pd.read_csv('test_data.csv')
test = pd.read_csv('test_public.csv')
# Internet和public数据共同特征总量训练
# 模型：

# 准备数据
X_train1 = train1_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train1 = train1_data['is_default']

X_train2 = train2_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train2 = train2_data['is_default']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])

X_test = test_data.drop(['earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)

# 缺失值填补
X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)

X_train_NN = (X_train.values).astype('float32') # all pixel values
y_train_NN = y_train.astype('int32')

X_test_NN = (X_test.values).astype('float32') # all pixel values


# 创建随机森林分类器对象
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练随机森林模型
rf.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = rf.predict(X_test)

# 将预测结果添加到测试集
test['is_default'] = y_pred

# 提取需要的列（loan_id和is_default）
submission = test[['loan_id', 'is_default']]

# 保存预测结果
submission.to_csv('submission3.csv', index=False)