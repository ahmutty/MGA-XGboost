import pandas as pd
import xgboost as xgb
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
        'subsample':0.8,
        'colsample_bytree':0.8,
        'min_child_weight':3,
        'alpha':1e-5,
        'seed':0,
        'nthread':4,
        'silent':1,
        'gamma':0.2,
        'learning_rate' : 0.01} 
watchlist = [(dtrain,'train')]
bst = xgb.train(params,dtrain,num_boost_round=5000,evals=watchlist)
bst.save_model('C:\\Users\\austl\Desktop\\baXgboost\\fanhua\\test_model2') # 保存实验模型
