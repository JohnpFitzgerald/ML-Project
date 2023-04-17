# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 01:50:20 2023

@author: Jfitz
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
#Load the data
file = '24HrFeatures.csv'
df = pd.read_csv(file)

# patient IDs for LeaveOneGroupOut
groups = df['patientID'].copy() 
# Define the features and target variable
X = df.copy().drop(['id','class','date','Category','counter','patientID'], axis=1)
y = df['class'].copy()

# Scale the features
#scaler = StandardScaler()
#scaler = PowerTransformer(method='yeo-johnson')
scaler = MinMaxScaler()

X[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X[['f.mean', 'f.sd', 'f.propZeros']])




# Define the models and their hyperparameters
rf_params = {'n_estimators': 1000, 
             'max_depth': 64, 
             'min_samples_split': 1, 
             'min_samples_leaf': 1,
             'max_features': 'sqrt', 
             'class_weight': 'balanced', 
             'random_state': 3023}
xgb_params = {'objective': 'multi:softmax', 
              'num_class': 3, 
              'max_depth': 64, 
              'learning_rate': 0.01, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'gamma': 2, 
              'random_state': 2020}
lgb_params = {'objective': 'multiclass', 
              'num_class': 3, 
              'num_leaves': 8, 
              'learning_rate': 0.01, 
              'max_depth': 4, 
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'reg_alpha': 2, 
              'random_state': 3020}

# Define the ensemble model
models = {'rf': RandomForestClassifier(**rf_params),
          'xgb': xgb.XGBClassifier(**xgb_params),
          'lgb': lgb.LGBMClassifier(**lgb_params)}

# Define the LOGO cross-validation object
logo = LeaveOneGroupOut()

# Initialize empty arrays to store the predicted labels and true labels for each fold
# for ensemble and individual models
pred_labels = np.array([])
true_labels = np.array([])
RFpred_labels = np.array([])
RFtrue_labels = np.array([])
XGpred_labels = np.array([])
XGtrue_labels = np.array([])
LGpred_labels = np.array([])
LGtrue_labels = np.array([])
# Loop over the folds of the LOGO cross-validation
for train_index, test_index in logo.split(X, y, groups=groups):
    # Split the data into training and test sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train each model on the training set for this fold
    for name, model in models.items():
        model.fit(X_train, y_train)

    # Make predictions on the test set for this fold using each model
    rf_pred = models['rf'].predict(X_test)
    xgb_pred = models['xgb'].predict(X_test)
    lgb_pred = models['lgb'].predict(X_test)

    RFpred_labels = np.concatenate([RFpred_labels, rf_pred ])
    RFtrue_labels = np.concatenate([RFtrue_labels, y_test])

    XGpred_labels = np.concatenate([XGpred_labels, xgb_pred])
    XGtrue_labels = np.concatenate([XGtrue_labels, y_test])
    
    LGpred_labels = np.concatenate([LGpred_labels, lgb_pred])
    LGtrue_labels = np.concatenate([LGtrue_labels, y_test])

    # Combine the predictions from all models using a simple voting scheme
    #ensemble_pred = np.round((xgb_pred + lgb_pred) / 3)
    ensemble_pred = np.round((rf_pred + xgb_pred + lgb_pred) / 3)
    # Add the predicted labels and true labels for this fold to the arrays
    pred_labels = np.concatenate([pred_labels, ensemble_pred])
    true_labels = np.concatenate([true_labels, y_test])


# Compute the accuracy of the RF model
RFaccuracy = accuracy_score(RFtrue_labels, RFpred_labels)
print("RF accuracy: {:.2f}%".format(RFaccuracy * 100))
# Compute the accuracy of the XGB model
XGaccuracy = accuracy_score(XGtrue_labels, XGpred_labels)
print("XGB accuracy: {:.2f}%".format(XGaccuracy * 100))
# Compute the accuracy of the LGB model
LGaccuracy = accuracy_score(LGtrue_labels, LGpred_labels)
print("LGB accuracy: {:.2f}%".format(LGaccuracy * 100))
# Compute the accuracy of the ensemble model
accuracy = accuracy_score(true_labels, pred_labels)
print("Ensemble accuracy: {:.2f}%".format(accuracy * 100))
