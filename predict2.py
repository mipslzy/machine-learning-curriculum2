import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.regularizers import l2
from tensorflow.contrib.metrics import streaming_auc
import pandas as pd
import tensorflow.contrib.keras as K



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

# 数据标准化
mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)
def standardize(x):
    return (x-mean_px)/std_px

# 缺失值填补
X_train.fillna(0, inplace = True)
X_test.fillna(0, inplace = True)

X_train_NN =(X_train - mean_px) / std_px
X_test_NN  = (X_test - mean_px) / std_px

X_train_NN = (X_train.values).astype('float32') # all pixel values
y_train_NN = y_train.astype('int32')

X_test_NN = (X_test.values).astype('float32') # all pixel values


# 模型构建
# 修改初始化、加归一层、加dropout、改用不同的metrics
seed = 43
np.random.seed(seed)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

input_shape = X_train_NN.shape[1]
b_size = 1024
max_epochs = 10

init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam(lr=0.001)

model = K.models.Sequential()
model.add(K.layers.Dense(units=256, input_dim=input_shape, kernel_initializer='he_normal', activation='relu',kernel_regularizer=l2(0.0001)))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(units= 64, kernel_initializer='he_normal', activation='relu'))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(units=1, kernel_initializer='he_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=simple_adam, metrics=['accuracy', auroc])

model.summary()

# 训练模型
print("Starting NN training")
h = model.fit(X_train_NN, y_train_NN, batch_size=b_size, epochs=max_epochs, shuffle=True, verbose=1)
print("NN training finished")

# 预测
pred_NN = model.predict(X_test_NN)
pred_NN = [item[0] for item in pred_NN]

# 预测结果
model.save('NN_model.h5')
submission = pd.DataFrame({'id':test['loan_id'], 'is_default':pred_NN})
submission.to_csv('submission2.csv', index = None)
