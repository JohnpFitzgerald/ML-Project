# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:07:57 2023

@author: fitzgeraldj
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
file = '24HrFeatures.csv'
#file = '4HrFeatures.csv'
df = pd.read_csv(file)


#Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID'],axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])


#Trainingset 10-fold cross validation
kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=2010)
# use leave-one-out cross-validation
logo = LeaveOneOut()     
#----------------------Logistic Regression


logReg = LogisticRegression()



logReg.fit(X_train, y_train)

# Evaluate the model's performance on the testing data
y_pred = logReg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f" Logistic Regression Accuracy: {accuracy}")
print(f"LR Confusion Matrix: \n{confusion_mat}")
print(f"LR Classification Report:\n{class_report}")


#--------------------- Random Forest

# create a random forest classifier object
rfc = RandomForestClassifier()

# fit the model on the training data
rfc.fit(X_train, y_train)

# predict on the test data
y_pred = rfc.predict(X_test)

# evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")
confusionRF = confusion_matrix(y_test, y_pred)
print(f"RanDom Forest Confusion Matrix: \n{confusionRF}")
class_report = classification_report(y_test, y_pred)
print(f"Random Forest Classification Report:\n{class_report}")


#----------------- XG Boost 


# Define the model
xgbm = xgb.XGBClassifier()

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
# feat_imp = pd.Series(xgbm.feature_importances_, index=X.columns).sort_values(ascending=False)
# plt.barh(feat_imp.index, feat_imp.values)
# plt.title("XGB Feature importances")
# plt.show()
# =============================================================================


#-----------------------Light GBM


# Define the model
lgbm = lgb.LGBMClassifier()

lgbm.fit(X_train, y_train)

# Evaluate the model
y_pred = lgbm.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Light GBM Accuracy score:", acc_score)
print("Light GBM Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"Light GBM Classification Report:\n{class_report}")
#feat_imp = pd.Series(lgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
#plt.barh(feat_imp.index, feat_imp.values)
#plt.title("Light GBM Feature importances")
#plt.show()


#------------------Decision Tree


#  Define the decision tree model
dtm = DecisionTreeClassifier()

dtm.fit(X_train, y_train)

# Evaluate the model
y_pred = dtm.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("Decision Tree Accuracy score:", acc_score)
print("Decision Tree Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"Decision Tree Classification Report:\n{class_report}")
#plt.figure(figsize=(10,6))
#plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Depressive', 'Control', 'Schizophrenic'])
#plt.savefig('DecisionTreeClassifier.pdf', dpi=300)
#plt.show()



models={'logReg':logReg,'rfc':rfc,'dtm':dtm,'lgbm':lgbm,'xgbm':xgbm,}
# Loop through the models and perform cross-validation
for model_name, model in models.items():
    # Use k-fold cross-validation
    print(f'{model_name} using k-fold cross-validation:')
    accuracy_scores = []
    for train_index, cv_index in kf.split(X_train,
                                          y_train):
        X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
        y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(f'Average accuracy kfold: {np.mean(accuracy_scores):.3f}')
    
    # Use leave-one-out cross-validation
    print(f'{model_name} using leave-one-out cross-validation:')
    accuracy_scores = []
    for train_index, cv_index in logo.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
        X_train, X_test = X.iloc[train_index], X.iloc[cv_index]
        y_train, y_test = y.iloc[train_index], y.iloc[cv_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(f'Average accuracy logo: {np.mean(accuracy_scores):.3f}')    