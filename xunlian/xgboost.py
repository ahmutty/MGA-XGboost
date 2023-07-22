
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data.csv')
pd.Series(data['Infectious Complications']).value_counts()
data.describe()
numerical_features = [x for x in data.columns if data[x].dtype == float]
category_features = [x for x in data.columns if data[x].dtype != float and x != 'Infectious Complications']
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
plt.title("Correlation analysis diagram")
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
from sklearn.model_selection import train_test_split
data_target_part = data['Infectious Complications']
data_features_part = data[[x for x in data.columns if x != 'Infectious Complications']]
x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.3, random_state = 2020)
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier()
clf.fit(x_train, y_train)
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)
y_pred = clf.predict(x_test) 
y_pred1= (y_pred >= 0.5)*1
from sklearn import metrics
print ('xgboost results：')
print ('AUC: %.4f' % metrics.roc_auc_score(y_test,y_pred))
print ('ACC: %.4f' % metrics.accuracy_score(y_test,y_pred1))
print ('Recall: %.4f' % metrics.recall_score(y_test,y_pred1))
print ('F1-score: %.4f' %metrics.f1_score(y_test,y_pred1))
print ('Precesion: %.4f' %metrics.precision_score(y_test,y_pred1))
metrics.confusion_matrix(y_test,y_pred1)
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train,train_predict))
print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))
confusion_matrix_result = metrics.confusion_matrix(test_predict,y_test)
print('The confusion matrix result:\n',confusion_matrix_result)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
clf.feature_importances_
#sns.barplot(x=data_features_part.columns, y=clf.feature_importances_,)
plt.figure(figsize = [20,10],dpi=100)
ax = sns.barplot(x=data_features_part.columns,y = clf.feature_importances_)
ax.set_xticklabels(labels = ['ASA Grading','Diabetes','Stoma','Operation Time','Tumor Location','Blood Transfusion','Chronic Lung Disease','LMR','NLR','PLR','LCR','PNI'],rotation = 45,fontsize = 18) # 放大横轴坐标并逆时针旋转45°
ax.set_yticklabels(labels = [0,0.025,0.05,0.075,0.100,0.125,0.150,0.175],fontsize = 18)
plt.xlabel('fea_name',fontsize=20)
plt.ylabel('fea_imp',fontsize=20)
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
