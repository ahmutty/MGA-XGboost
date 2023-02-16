import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
# 读取数据预处理的训练集数据
df = pd.read_csv('data.csv')
df.info()
df.head()
pd.Series(df['Infectious Complications']).value_counts()
numerical_features = [x for x in df.columns if df[x].dtype == float]
category_features = [x for x in df.columns if df[x].dtype != float and x != 'Infectious Complications']
def get_mapfunction(x):
    mapp = dict(zip(x.unique().tolist(),
         range(len(x.unique().tolist()))))
    def mapfunction(y):
        if y in mapp:
            return mapp[y]
        else:
            return -1
    return mapfunction
for i in category_features:
    df[i] = df[i].apply(get_mapfunction(df[i]))
data_target_part = df['Infectious Complications']
data_features_part = df[[x for x in df.columns if x != 'Infectious Complications']]
## 测试集大小为30%， 70%/30%分
x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.3, random_state = 2020)
dtest = xgb.DMatrix(x_test)
bst1 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model1')
bst2 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model2')
bst3 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model3')
bst4 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model4')
bst5 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model5')
bst6 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model6')
bst7 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model7')
bst8 = xgb.Booster(model_file='C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model8')
ypred1 = bst1.predict(dtest)
ypred2 = bst2.predict(dtest)
ypred3 = bst3.predict(dtest)
ypred4 = bst4.predict(dtest)
ypred5 = bst5.predict(dtest)
ypred6 = bst6.predict(dtest)
ypred7 = bst7.predict(dtest)
ypred8 = bst8.predict(dtest)
ypred = 0.148*ypred1 + 0.038*ypred2 + 0.148*ypred3 + 0.074*ypred4 + 0.148*ypred5 + 0.074*ypred6 + 0.074*ypred7 + 0.296*ypred8
y_pred = (ypred >= 0.5)*1
y_pred13 = (ypred2 >= 0.5)*1#xg
#贪心算法后的xgboost
from sklearn import metrics
print ('适度贪婪xgboost的结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
#线性回归
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
LR = LinearRegression()
LR.fit(x_train, y_train)  # 训练
y_pred2 = LR.predict(x_test)  # 预测
y_pred3 = (y_pred2 >= 0.5)*1
print ('线性回归的试验结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred2))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred3))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred3))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred3))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred3))
metrics.confusion_matrix(y_test,y_pred3)
rmse1 = mean_absolute_error(y_test, y_pred3)
rmse_scores1 = []
rmse_scores1.append(rmse1)
print("rmse scores : ", rmse_scores1)
print(f'average rmse scores : {np.mean(rmse_scores1)}')
#随机森林
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()  # 基模型
RFR.fit(x_train, y_train)
y_pred4 = RFR.predict(x_test)
y_pred5= (y_pred4 >= 0.5)*1#数据转换
print ('随机森林结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred4))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred5))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred5))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred5))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred5))
metrics.confusion_matrix(y_test,y_pred5)
rmse2 = mean_absolute_error(y_test, y_pred5)
rmse_scores2 = []
rmse_scores2.append(rmse2)
print("rmse scores : ", rmse_scores2)
print(f'average rmse scores : {np.mean(rmse_scores2)}')
#lightGBM
import lightgbm as lgb
LGBR = lgb.LGBMRegressor()  # 基模型
LGBR.fit(x_train, y_train)
y_pred6 = LGBR.predict(x_test)
y_pred7= (y_pred6 >= 0.5)*1#数据转换
print ('lightGBM结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred6))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred7))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred7))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred7))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred7))
metrics.confusion_matrix(y_test,y_pred7)
rmse3 = mean_absolute_error(y_test, y_pred7)
rmse_scores3 = []
rmse_scores3.append(rmse3)
print("rmse scores : ", rmse_scores3)
print(f'average rmse scores : {np.mean(rmse_scores3)}')
#支持向量机
from sklearn.svm import SVR
clf = SVR()
rf = clf.fit (x_train, y_train.ravel())
y_pred8 = rf.predict(x_test)
y_pred9= (y_pred8 >= 0.5)*1#数据转换
print ('SVM结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred8))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred9))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred9))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred9))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred9))
metrics.confusion_matrix(y_test,y_pred9)
rmse4 = mean_absolute_error(y_test, y_pred9)
rmse_scores4 = []
rmse_scores4.append(rmse4)
print("rmse scores : ", rmse_scores4)
print(f'average rmse scores : {np.mean(rmse_scores4)}')
#神经网络
from sklearn.neural_network import MLPRegressor
clf2 = MLPRegressor()
rf2= clf2.fit (x_train, y_train.ravel())
y_pred10 = rf2.predict(x_test)
y_pred11= (y_pred10 >= 0.5)*1#数据转换
print ('神经网络结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred10))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred11))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred11))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred11))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred11))
metrics.confusion_matrix(y_test,y_pred11)
rmse5 = mean_absolute_error(y_test, y_pred11)
rmse_scores5 = []
rmse_scores5.append(rmse5)
print("rmse scores : ", rmse_scores5)
print(f'average rmse scores : {np.mean(rmse_scores5)}')
#绘制auc曲线
fpr,tpr,threshold = roc_curve(y_test, ypred) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值

fpr1,tpr1,threshold = roc_curve(y_test, y_pred2) ###计算真正率和假正率
roc_auc1 = auc(fpr1,tpr1) ###计算auc的值

fpr2,tpr2,threshold = roc_curve(y_test, y_pred4) ###计算真正率和假正率
roc_auc2 = auc(fpr2,tpr2) ###计算auc的值

fpr3,tpr3,threshold = roc_curve(y_test, y_pred6) ###计算真正率和假正率
roc_auc3 = auc(fpr3,tpr3) ###计算auc的值
plt.figure()

fpr4,tpr4,threshold = roc_curve(y_test, y_pred8) ###计算真正率和假正率
roc_auc4 = auc(fpr4,tpr4) ###计算auc的值
plt.figure()

fpr5,tpr5,threshold = roc_curve(y_test, y_pred10) ###计算真正率和假正率
roc_auc5 = auc(fpr5,tpr5) ###计算auc的值
plt.figure()

fpr6,tpr6,threshold = roc_curve(y_test, ypred2) ###计算真正率和假正率
roc_auc6 = auc(fpr6,tpr6) ###计算auc的值
plt.figure()
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(20,15),dpi=300)
plt.plot(fpr, tpr, color='#32AEEC', lw=4, label='MGA-XGBoost (AUC = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr6, tpr6, color='#E3863C', lw=4, label='XGBoost (AUC = %0.3f)' % roc_auc6) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr3, tpr3, color='#E9262E', lw=4, label='LGBM (AUC = %0.3f)' % roc_auc3) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr2, tpr2, color='#444A9B', lw=4, label='Random Forest (AUC = %0.3f)' % roc_auc2) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr1, tpr1, color='#B49D30', lw=4, label='Linear Regression (AUC = %0.3f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr5, tpr5, color='#030100', lw=4, label='SVM (AUC = %0.3f)' % roc_auc5) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr4, tpr4, color='#712A7D', lw=4, label='BP (AUC = %0.3f)' % roc_auc4) ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='grey', lw=4, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xticks(fontproperties = 'Times New Roman', size = 25)
plt.yticks(fontproperties = 'Times New Roman', size = 25)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve',fontsize=30)
plt.legend(loc='lower right',fontsize=20)


plt.savefig( 'roc.png')
plt.show()
