# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:27:38 2023

@author: Jfitz
"""
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
color = sns.color_palette()
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

#file = '4HourlyFeatures.csv'
file = '24HrFeatures.csv'

#data = pd.read_csv(current_path + file)
#JFitz - Set to read file from same directory as code
df = pd.read_csv(file)

# Calculate kurtosis for each feature column and add as new column to dataframe
kurt_list = []
for col in df.columns:
    if col in ['f.mean','f.sd','f.propZeros']:
       kurt = df[col].kurtosis()
       df[col] = [kurt]

# Step 3: Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID'],axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Encode the categorical variable
#X_train = pd.get_dummies(X_train, columns=['class1'])
#X_test = pd.get_dummies(X_test, columns=['class1'])

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros', 'kurtosis']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros', 'kurtosis']])
X_test[['f.mean', 'f.sd', 'f.propZeros', 'kurtosis']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros', 'kurtosis']])



#-----------------------Light GBM

params = {
     'objective': 'multiclass',
     'metric':'multi_logloss',
     'num_class': 3,
     'learning_rate': 0.1,
     'num_leaves': 555, 
     'max_depth': 2,
    # 'lambda_l1': 0.01,
    # 'lambda_l2': 0.01,     
     'n_estimators': 130,
     #'min_Data_in_leaf':50,
     #'feature_fraction': 0.9,
     #'bagging_fraction':0.9,
     #'min_child_samples':20,
     #'min_split_gain':0.1,
     #'max_bin':255,
     #'early_stopping_rounds':10
 }

param_grid = {
    'num_leaves':[31,63,127],
    'max_depth':[-1,2,5,10],
    'learning_rate':[0.05,0.1,0.3],
    'feature_fraction':[0.5,0.7,0.9],
    'bagging_fraction':[0.5,0.7,0.9]
    
    }
params_LGB = {
    'task': 'train',
    'num_class':3,
    #'boosting': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth':2,
    'num_leaves': 555,
    'learning_rate': 0.1,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'bagging_seed': 2018,
  #  'verbose': 0,
   # 'num_threads':16
}
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
feat_imp = pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.barh(feat_imp.index, feat_imp.values)
plt.title("Light GBM Feature importances")
plt.show()   


#Train and evaluate the model using kfold cross-validation
# =============================================================================
# accuracy_scores = []
# for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
#                                           y_train.ravel()):
#     X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
#     lgbm.fit(X_train, y_train)
#     y_pred = lgbm.predict(X_test)
#     accuracy_scores.append(accuracy_score(y_test, y_pred))
# 
# print(f"LightGBM Accuracy scores: {accuracy_scores}")
# print(f"Light GBM Mean accuracy score: {np.mean(accuracy_scores)}")
# 
# =============================================================================

#----------------- XG Boost 

params={ 
        'booster':'gbtree',
        'objective': 'multi:softmax',
        'num_class':3,
        'max_depth':5,
        'learning_rate':0.1,
        'n_estimators':100,
        'subsample':0.7,
        'colsample_bytree':0.9,
        'min_child_weight': 1,
        'gamma':1,
        'lambda':0.5,
        'alpha':1,
        'max_delta_step':0.5,
        'eta':0.1,
        'colsample_bynode':0.7
        }

params_xGB = {
    'nthread':16,
    'learning_rate': 0.1,
    'gamma': 1,
    'max_depth': 5,
    'min_child_weight': 1,
    'max_delta_step': 0.5,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'objective':'multi:softmax',
    'num_class':1,
    'eval_metric':'logloss',
    'seed':2018
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
feat_imp = pd.Series(xgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.barh(feat_imp.index, feat_imp.values)
plt.title("XGB Feature importances")
plt.show()
# 
# =============================================================================
#Train and evaluate the model using leave one patient out cross-validation
# =============================================================================
# accuracy_scores = []
# for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
#                                           y_train.ravel()):
#     X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
#     lgbm.fit(X_train, y_train)
#     y_pred = xgbm.predict(X_test)
#     accuracy_scores.append(accuracy_score(y_test, y_pred))
# 
# print(f"XGB Accuracy scores: {accuracy_scores}")
# print(f"XGB Mean accuracy score: {np.mean(accuracy_scores)}")
# =============================================================================

    

