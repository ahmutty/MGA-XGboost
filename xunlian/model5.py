from pylab import mpl 
import pandas as pd
from xgboost import plot_importance
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# 读取数据预处理的训练集数据：
df = pd.read_csv('data.csv')
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
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test)
#booster:
params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':7,
        'lambda':1,
        'subsample':0.9,
        'colsample_bytree':0.95,
        'min_child_weight':4,
        'alpha':1e-5,
        'seed':0,
        'nthread':4,
        'silent':1,
        'gamma':0,
        'learning_rate' : 0.01} #0.8319/0.8299--0.8302 num-boost_round = 4000/7000
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
bst.save_model('C:\\Users\\austl\\Desktop\\baXgboost\\xunlian\\model5') # 保存实验模型
ypred=bst.predict(dtest)
y_pred = (ypred >= 0.5)*1
# 画出特征得分图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 使显示图标自适应
plt.rcParams['figure.autolayout'] = True
plot_importance(bst)
# 画出AUC
from sklearn import metrics
print ('参数模型5下的实验结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred))
metrics.confusion_matrix(y_test,y_pred)
fpr,tpr,threshold = roc_curve(y_test, ypred) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example(parameters5)')
plt.legend(loc="lower right")
plt.show()