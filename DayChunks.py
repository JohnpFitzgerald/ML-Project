# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:39:09 2023

@author: fitzgeraldj
"""

import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#file = '24HrFeatures.csv'
file = '4HrFeatures.csv'
df = pd.read_csv(file)

#options = ['12-16:00','16-20:00']
	
# selecting rows based on condition
#df = df[df['segment'].isin(options)]
	

#Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID','segment'], axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])


X_train, X_test, y_train, y_test = train_test_split(X,
                                    y, test_size=0.25,
                                    random_state=2550, stratify=y)

#Trainingset 10-fold cross validation
k_fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2010)

#Create the models:

#-----------------------Light GBM

params = {
     #'boosting': 'dart',
     'objective': 'multiclass',
     'metric':'multi_logloss',
     'num_class': 3,
     'learning_rate': 0.1,
     #'num_iterations': 55,
     'num_leaves': 5, 
     'max_depth': 5,
     #'lambda_l1': 0.01,
     #'lambda_l2': 0.01,     
     'n_estimators': 100,
    # 'min_Data_in_leaf':50,
     #'feature_fraction': 0.1,
     'subsample':0.75,    
     #'bagging_fraction':0.9,
     'min_child_samples':5,
     'min_split_gain':0.8,
     'max_bin':255,
    # 'dart':10
     #'early_stopping_rounds':10
 }

# =============================================================================
# param_grid = {
#     'num_leaves':[31,63,127],
#     'max_depth':[-1,2,5,10],
#     'learning_rate':[0.05],
#     'feature_fraction':[0.5,0.7,0.9],
#     'bagging_fraction':[0.5,0.7,0.9]
#     
#     }
# params_LGB = {
#     'task': 'train',
#     'num_class':3,
#     #'boosting': 'gbdt',
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'max_depth':2,
#     'num_leaves': 555,
#     'learning_rate': 0.1,
#     'feature_fraction': 1.0,
#     'bagging_fraction': 1.0,
#     'bagging_freq': 0,
#     'bagging_seed': 2018,
#   #  'verbose': 0,
#    # 'num_threads':16
# }
# 
# SEARCH_PARAMS = {'learning_rate': 0.1,
#                 'max_depth': 15,
#                 'num_leaves': 10,
#                 'feature_fraction': 0.8,
#                 'subsample': 0.2}
# 
# FIXED_PARAMS={'objective': 'multiclass',
#              'metric': 'multi_logloss',
#              'is_unbalance':True,
#              'bagging_freq':5,
#              #'boosting':'dart',
#              'num_boost_round':20,
#              'early_stopping_rounds':30}
# 
# def train_evaluate(search_params):
#    # you can download the dataset from this link(https://www.kaggle.com/c/santander-customer-transaction-prediction/data)
#    # import Dataset to play with it
#    data= pd.read_csv("24HrFeatures.csv")
#    X = data.copy().drop(['id','class','date','Category','counter','patientID'], axis=1)
#    y = data['class'].copy()
#    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2023)
#    train_data = lgb.Dataset(X_train, label=y_train)
#    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
# 
#    params = {'metric':FIXED_PARAMS['metric'],
#              'objective':FIXED_PARAMS['objective'],
#              **search_params}
# 
#    model = lgb.train(params, train_data,
#                      valid_sets=[valid_data],
#                      num_boost_round=FIXED_PARAMS['num_boost_round'],
#                      #early_stopping_rounds=FIXED_PARAMS['early_stopping_rounds'],
#                      valid_names=['valid'])
#    score = model.best_score['valid']
#    return score
# 
# print(train_evaluate({
#      #'boosting': 'dart',
#      'objective': 'multiclass',
#      #'metric':'multi_logloss',
#      'num_class': 3,
#      #'learning_rate': 0.1,
#      #'num_leaves': 10, 
#      #'max_depth': 10,    
#      #'n_estimators': 30,
#      'min_child_samples':5,
#      'min_split_gain':0.8,
#      'max_bin':255,
#      }
# ))
# =============================================================================
#lgb_model = lgb.LGBMClassifier(**params)

lgbm = lgb.LGBMClassifier(**params)
#grid = GridSearchCV(lgbm, param_grid)
#grid.fit(X_train, y_train)

lgbm.fit(X_train, y_train)

# Evaluate the model
y_pred = lgbm.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Light GBM Accuracy score:", acc_score)
print("Light GBM Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"Light GBM Classification Report:\n{class_report}")

# =============================================================================
# feat_imp = pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
# plt.barh(feat_imp.index, feat_imp.values)
# plt.title("Light GBM Feature importances")
# plt.show()   
# 
# =============================================================================
#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
    y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"LightGBM Accuracy scores: {accuracy_scores}")
print(f"Light GBM Mean accuracy score: {np.mean(accuracy_scores)}")


#----------------- XG Boost 

params={       
        'booster':'gbtree',
        'objective': 'multi:softmax',
        'eval_metric':'logloss',
        'num_class':3,
        'max_depth':5,
        'learning_rate':0.2,
        'n_estimators':1500,
        'subsample':0.9,
        'colsample_bytree':0.9,
        'min_child_weight': 1,
        'gamma':1,
        'lambda':0.1,
        'alpha':0.5,
        'max_delta_step':1,
        'eta':0.1,
        'colsample_bynode':0.3
        }


xgbm = xgb.XGBClassifier(**params)


xgbm.fit(X_train, y_train)

# Evaluate the model
y_pred = xgbm.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("XGB Accuracy score:", acc_score)
print("XGB Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"XGB Classification Report:\n{class_report}")
# =============================================================================
# # =============================================================================
# feat_imp = pd.Series(xgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
# plt.barh(feat_imp.index, feat_imp.values)
# plt.title("XGB Feature importances")
# plt.show()
# # =============================================================================
# =============================================================================
#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
    y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
    lgbm.fit(X_train, y_train)
    y_pred = xgbm.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"XGB Accuracy scores: {accuracy_scores}")
print(f"XGB Mean accuracy score: {np.mean(accuracy_scores)}")

    
