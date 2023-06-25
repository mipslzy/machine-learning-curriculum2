import lightgbm
import pandas as pd

# 加载数据
train1_data = pd.read_csv('train1_data.csv')
train2_data = pd.read_csv('train2_data.csv')
test_data = pd.read_csv('test_data.csv')
test = pd.read_csv('test_public.csv')
# Internet和public数据共同特征总量训练
# 模型：梯度提升决策树 （LigthGBM）

# 准备数据
X_train1 = train1_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train1 = train1_data['is_default']

X_train2 = train2_data.drop(['is_default','earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)
y_train2 = train2_data['is_default']

X_train = pd.concat([X_train1, X_train2])
y_train = pd.concat([y_train1, y_train2])

X_test = test_data.drop(['earlies_credit_mon','loan_id','user_id'], axis = 1, inplace = False)

# 模型训练
clf_ex=lightgbm.LGBMRegressor(n_estimators = 200)
clf_ex.fit(X = X_train, y = y_train)
clf_ex.booster_.save_model('LGBMmode.txt')

# 预测
pred = clf_ex.predict(X_test)

# 预测结果
submission = pd.DataFrame({'id':test['loan_id'], 'is_default':pred})
submission.to_csv('submission.csv', index = None)