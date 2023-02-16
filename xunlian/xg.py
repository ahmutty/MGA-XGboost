##  基础函数库
import numpy as np
import pandas as pd
## 绘图函数库
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data.csv')
pd.Series(data['Infectious Complications']).value_counts()
data.describe()
numerical_features = [x for x in data.columns if data[x].dtype == float]
category_features = [x for x in data.columns if data[x].dtype != float and x != 'Infectious Complications']
## 选取特征与标签组合的散点可视化
sns.pairplot(data=data[['ASA Grading','Diabetes','Stoma','Operation Time','Tumor Location','Blood Transfusion','Chronic Lung Disease','LMR','NLR','PLR','LCR','PNI'] + ['Infectious Complications']], diag_kind='hist', hue= 'Infectious Complications')
plt.show()
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
for col in data[numerical_features].columns:
    if col != 'Infectious Complications':
        sns.boxplot(x='Infectious Complications', y=col, saturation=0.5, palette='pastel', data=data)
        plt.title(col)
        plt.show()
plt.figure(figsize=(20, 20))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.title("相关性分析图")




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
    data[i] = data[i].apply(get_mapfunction(data[i]))
## 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split
## 选择其类别为0和1的样本 （不包括类别为2的样本）
data_target_part = data['Infectious Complications']
data_features_part = data[[x for x in data.columns if x != 'Infectious Complications']]
## 测试集大小为30%， 70%/30%分
x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.3, random_state = 2020)
## 导入XGBoost模型
from xgboost.sklearn import XGBClassifier
## 定义 XGBoost模型
clf = XGBClassifier()
# 在训练集上训练XGBoost模型
clf.fit(x_train, y_train)
## 在训练集和测试集上分布利用训练好的模型进行预测
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
y_pred = clf.predict(x_test)  # 预测
y_pred1= (y_pred >= 0.5)*1
from sklearn import metrics
print ('xgboost的结果：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred1))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred1))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred1))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred1))
metrics.confusion_matrix(y_test,y_pred1)
## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))
## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
# 利用热力图对于结果进行可视化
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
# XGBoost的特征选择属于特征选择中的嵌入式方法，在XGboost中可以用属性feature_importances_去查看特征的重要度。
clf.feature_importances_
#sns.barplot(x=data_features_part.columns, y=clf.feature_importances_,)
plt.figure(figsize = [20,10],dpi=100)
ax = sns.barplot(x=data_features_part.columns,y = clf.feature_importances_)
ax.set_xticklabels(labels = ['ASA Grading','Diabetes','Stoma','Operation Time','Tumor Location','Blood Transfusion','Chronic Lung Disease','LMR','NLR','PLR','LCR','PNI'],rotation = 45,fontsize = 18) # 放大横轴坐标并逆时针旋转45°
ax.set_yticklabels(labels = [0,0.025,0.05,0.075,0.100,0.125,0.150,0.175],fontsize = 18) # 放大纵轴坐标
plt.xlabel('fea_name',fontsize=20) # 放大横轴名称
plt.ylabel('fea_imp',fontsize=20) # 放大纵轴名称

from sklearn.metrics import accuracy_score
from xgboost import plot_importance
def estimate(model, data):
    # sns.barplot(data.columns,model.feature_importances_)
    ax1 = plot_importance(model, importance_type="gain")
    ax1.set_title('gain')
    ax2 = plot_importance(model, importance_type="weight")
    ax2.set_title('weight')
    ax3 = plot_importance(model, importance_type="cover")
    ax3.set_title('cover')
    plt.show()
def classes(data, label, test):
    model = XGBClassifier()
    model.fit(data, label)
    ans = model.predict(test)
    estimate(model, data)
    return ans
ans = classes(x_train, y_train, x_test)
pre = accuracy_score(y_test, ans)
print('acc=', accuracy_score(y_test, ans))

from sklearn.model_selection import GridSearchCV
# 定义参数取值范围
learning_rate = [0.1, 0.3, 0.6]
subsample = [0.8, 0.9]
colsample_bytree = [0.6, 0.8]
max_depth = [3, 6, 8]
parameters = {
    'learning_rate': learning_rate,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'max_depth': max_depth
}
model = XGBClassifier(n_estimators =20)

clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
clf = clf.fit(x_train, y_train)
clf.best_params_

clf = XGBClassifier(colsample_bytree = 0.7, learning_rate = 0.1, max_depth = 6, subsample = 0.7)

clf.fit(x_train, y_train)
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))

confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
print('The confusion matrix result:\n', confusion_matrix_result)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

