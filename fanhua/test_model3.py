import pandas as pd
import xgboost as xgb

# 读取数据预处理的训练集数据：33464*6746
df = pd.read_csv('sj2.csv')
train_x = df.loc[:,['ASA Grading','Diabetes','Stoma','chronic lung disease','Operation Time','Tumor Location','Blood Transfusion','NLR','PLR','LCR','PNI']]
train_y = df.loc[:,['Infectious Complications']]
dtrain=xgb.DMatrix(train_x,label=train_y)
#booster:
params={'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':3,
        'lambda':1,
        'subsample':0.9,
        'colsample_bytree':0.5,
        'min_child_weight':3,
        'alpha':1e-5,
        'seed':0,
        'nthread':4,
        'silent':1,
        'gamma':0.4,
        'learning_rate' : 0.01} 
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=6000,evals=watchlist)
bst.save_model('C:\\Users\\austl\Desktop\\baXgboost\\fanhua\\test_model3') # 保存实验模型
