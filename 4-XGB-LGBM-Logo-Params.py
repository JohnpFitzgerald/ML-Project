# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:19:53 2023

@author: fitzgeraldj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:45:26 2023

@author: fitzgeraldj
"""

import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
file = '24HrFeatures.csv'
#file = '4HrFeatures.csv'
df = pd.read_csv(file)


#Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID'], axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, stratify=y)

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])


#X_train, X_test, y_train, y_test = train_test_split(X,
    #                                y, test_size=0.18,
        #                            stratify=y)

#Trainingset 10-fold cross validation
kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=2010)
# use leave-one-out cross-validation
logo = LeaveOneOut() 


#Create the 2 models:

#1.-----------------------Light GBM

params = {
     #'boosting': 'dart',
     'objective': 'multiclass',
     'metric':'multi_logloss',
     'num_class': 3,
     'learning_rate': 0.05,
    # 'num_leaves':127, 
     #'feature_fraction':0.5,
     #'bagging_fraction':0.7,
     'max_depth': 2,    
     #'n_estimators': 100,
     'subsample':1,    
     #'min_child_samples':5,
     #'min_split_gain':0.8,
    # 'max_bin':25
 }

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

#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, cv_index in logo.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
    y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"LightGBM LOGO Accuracy scores: {accuracy_scores}")
print(f"Light GBM Mean LOGO accuracy score: {np.mean(accuracy_scores)}")


#2.----------------- XG Boost 

params={       
     #   'booster':'gbtree',
        'objective': 'multi:softmax',
        'eval_metric':'logloss',
        'num_class':3,
        'max_depth':5,
        'learning_rate':0.8,
        'n_estimators':3500,
        'subsample':0.4,
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
feat_imp = pd.Series(xgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.barh(feat_imp.index, feat_imp.values)
plt.title("XGB Feature importances")
plt.show()
# =============================================================================
#Train and evaluate the model using leave one patient out cross-validation
accuracy_scores = []
for train_index, cv_index in logo.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
    y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
    lgbm.fit(X_train, y_train)
    y_pred = xgbm.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(f"XGB LOGO Accuracy scores: {accuracy_scores}")
print(f"XGB LOGO Mean accuracy score: {np.mean(accuracy_scores)}")
