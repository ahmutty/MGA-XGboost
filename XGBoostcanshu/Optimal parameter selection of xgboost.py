import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV  # Perforing grid search
df = pd.read_csv('sj2.csv')
train = df
target = 'Infectious Complications'
data_target_part = df['Infectious Complications']
data_features_part = df[[x for x in df.columns if x != 'Infectious Complications']]
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Infectious Complications'], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Infectious Complications'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Infectious Complications'], dtrain_predprob))
# Choose all predictors except target & IDcols
A = [x for x in train.columns if x not in [target]]
predictors = [x for x in train.columns if x not in [target]]
xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=140,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb1, train, predictors)
# Firstly, the parameters of max _ depth and min _ child _ weight are adjusted.
param_test1 = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 2, 3, 4, 5, 6]
}
grid1 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=140,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test1,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
grid1.fit(train[predictors], train[target])
grid1.cv_results_, grid1.best_params_, grid1.best_score_
print("grid1.cv_results_", grid1.cv_results_,
      "grid1.best_params_", grid1.best_params_,
      "grid1.best_score_:", grid1.best_score_)
param_test2 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch2 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=3,
    min_child_weight=3,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test2,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch2.fit(train[predictors], train[target])
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
print("gsearch2.cv_results_:", gsearch2.cv_results_,
      " gsearch2.best_params_:", gsearch2.best_params_,
      "gsearch2.best_score_:", gsearch2.best_score_)
param_test3 = {
    'subsample': [i / 10.0 for i in range(1, 10, 2)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 2)]
}
gsearch3 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test3,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch3.fit(train[predictors], train[target])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
print("gsearch3.cv_results_:", gsearch3.cv_results_,
      " gsearch3.best_params_:", gsearch3.best_params_,
      "gsearch3.best_score_:", gsearch3.best_score_)
# Results subsample = 0.9.colsample _ bytree = 0.5
# is more accurate, with a scale value of 0.05
param_test4 = {
    'subsample': [i / 10.0 for i in range(8, 10)],
    'colsample_bytree': [i / 10.0 for i in range(4, 7)]
}
gsearch4 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test4,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch4.fit(train[predictors], train[target])
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
print("gsearch4.cv_results_", gsearch4.cv_results_,
      "gsearch4.best_params_:", gsearch4.best_params_,
      "gsearch4.best_score_:", gsearch4.best_score_)
param_test5 = {
    'subsample': [i / 10.0 for i in range(1, 10, 2)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 2)]
}
gsearch5 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=7,
    min_child_weight=5,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test5,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch5.fit(train[predictors], train[target])
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_
print("gsearch5.cv_results_:", gsearch5.cv_results_,
      " gsearch5.best_params_:", gsearch5.best_params_,
      "gsearch5.best_score_:", gsearch5.best_score_)
# The test result is optimal : subsample = 0.9, colsample _ bytree = 0.5 
# Suboptimal : subsample = 0.95, colsample _ bytree = 0.5
# precision
param_test6 = {
    'subsample': [i / 10.0 for i in range(8, 10)],
    'colsample_bytree': [i / 10.0 for i in range(4, 7)]
}
gsearch6 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=3,
    min_child_weight=3,
    gamma=0.,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test6,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch6.fit(train[predictors], train[target])
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_
print("gsearch6.cv_results_:", gsearch6.cv_results_,
      "gsearch6.best_params_:", gsearch6.best_params_,
      "gsearch6.best_score_:", gsearch6.best_score_)
param_test7 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch7 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=7,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test7,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch7.fit(train[predictors], train[target])
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_
print("gsearch7.cv_results_:", gsearch7.cv_results_,
      " gsearch7.best_params_:", gsearch7.best_params_,
      "gsearch7.best_score_:", gsearch7.best_score_)
param_test8 = {
    'subsample': [i / 10.0 for i in range(1, 10, 2)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 2)]
}
gsearch8 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=7,
    min_child_weight=4,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test8,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch8.fit(train[predictors], train[target])
gsearch8.cv_results_, gsearch8.best_params_, gsearch8.best_score_
print("gsearch8.cv_results_:", gsearch8.cv_results_,
      " gsearch8.best_params_:", gsearch8.best_params_,
      "gsearch8.best_score_:", gsearch8.best_score_)
param_test9 = {
    'subsample': [0.85, 0.9, 0.95],
    'colsample_bytree': [0.85, 0.9, 0.95]
}
gsearch9 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=7,
    min_child_weight=4,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test9,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch9.fit(train[predictors], train[target])
gsearch9.cv_results_, gsearch9.best_params_, gsearch9.best_score_
print("gsearch9.cv_results_:", gsearch9.cv_results_,
      " gsearch9.best_params_:", gsearch9.best_params_,
      "gsearch9.best_score_:", gsearch9.best_score_)
param_test10 = {
    'subsample': [i / 10.0 for i in range(1, 10, 2)],
    'colsample_bytree': [i / 10.0 for i in range(1, 10, 2)]
}
gsearch10 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=7,
    min_child_weight=4,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test10,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch10.fit(train[predictors], train[target])
gsearch10.cv_results_, gsearch10.best_params_, gsearch10.best_score_
print("gsearch10.cv_results_:", gsearch10.cv_results_,
      "gsearch10.best_params_:", gsearch10.best_params_,
      "gsearch10.best_score_:", gsearch10.best_score_)
param_test11 = {
    'subsample': [0.85, 0.9, 0.95],
    'colsample_bytree': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
}
gsearch11 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=7,
    min_child_weight=4,
    gamma=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test11,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch11.fit(train[predictors], train[target])
gsearch11.cv_results_, gsearch11.best_params_, gsearch11.best_score_
print("gsearch11.cv_results_:", gsearch11.cv_results_,
      "gsearch11.best_params_:", gsearch11.best_params_,
      "gsearch11.best_score_:", gsearch11.best_score_)
# Get the first set of parameters : max _ depth = 3, min _ child _ weight = 3, gamma = 0.2, subsample = 0.9, colsample _ bytree = 0.8. 
# Get the second set of parameters : max _ depth = 3, min _ child _ weight = 3, gamma = 0.2, subsample = 0.8, colsample _ bytree = 0.8. 
# Get the third set of parameters : max _ depth = 3, min _ child _ weight = 3, gamma = 0.4, subsample = 0.9, colsample _ bytree = 0.5. 
# Get the fourth set of parameters : max _ depth = 3, min _ child _ weight = 3, gamma = 0.4, subsample = 0.95, colsample _ bytree = 0.5. 
# Get the fifth set of parameters : max _ depth = 7, min _ child _ weight = 4, gamma = 0, subsample = 0.9, colsample _ bytree = 0.95. 
#Get the sixth set of parameters : max _ depth = 7, min _ child _ weight = 4, gamma = 0, subsample = 0.95, colsample _ bytree = 0.95. 
# Get the seventh set of parameters : max _ depth = 7, min _ child _ weight = 4, gamma = 0.1, subsample = 0.9, colsample _ bytree =0.15
# Get the eighth set of parameters : max _ depth = 7, min _ child _ weight = 4, gamma = 0.1, subsample = 0.9, colsample _ bytree = 0.3.
param_test12 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
gsearch12 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test12,
    scoring='roc_auc',
    n_jobs=4, cv=5)
gsearch12.fit(train[predictors], train[target])
gsearch12.cv_results_, gsearch12.best_params_, gsearch12.best_score_
print("gsearch12.cv_results_:", gsearch12.cv_results_,
      " gsearch12.best_params_:", gsearch12.best_params_,
      "gsearch12.best_score_:", gsearch12.best_score_)
# reg_alpha=0.00001     Continue debugging
param_test13 = {
    'reg_alpha': [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}
gsearch13 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test13,
    scoring='roc_auc',
    cv=5)
gsearch13.fit(train[predictors], train[target])
gsearch13.cv_results_, gsearch13.best_params_, gsearch13.best_score_
print("gsearch13.cv_results_:", gsearch13.cv_results_,
      "gsearch13.best_params_:", gsearch13.best_params_,
      "gsearch13.best_score_:", gsearch13.best_score_)
param_test14 = {
    'reg_lambda': [0, 1e-6, 1e-5, 1e-4, 1e-3]
}
gsearch14 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    reg_alpha=1e-5,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test14,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch14.fit(train[predictors], train[target])
gsearch14.cv_results_, gsearch14.best_params_, gsearch14.best_score_
print("gsearch14.cv_results_", gsearch14.cv_results_,
      "gsearch14.best_params_:", gsearch14.best_params_,
      "gsearch14.best_score_:", gsearch14.best_score_)
param_test15 = {
    'reg_lambda': [1, 10, 100]
}
gsearch15 = GridSearchCV(estimator=XGBClassifier(
    learning_rate=0.1,
    n_estimators=177,
    max_depth=3,
    min_child_weight=3,
    gamma=0.2,
    reg_alpha=1e-5,
    subsample=0.9,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27),
    param_grid=param_test15,
    scoring='roc_auc',
    n_jobs=4,
    cv=5)
gsearch15.fit(train[predictors], train[target])
gsearch15.cv_results_, gsearch15.best_params_, gsearch15.best_score_
print("gsearch15.cv_results_:", gsearch15.cv_results_,
      "gsearch15.best_params_:", gsearch15.best_params_,
      "gsearch15.best_score_:", gsearch15.best_score_)
