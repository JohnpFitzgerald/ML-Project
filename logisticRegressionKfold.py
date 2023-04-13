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

file = '24HrFeatures.csv'
#file = '4HrFeatures.csv'
df = pd.read_csv(file)


#Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID'], axis=1)
y = df['class'].copy()

# assuming X and y are the feature and target matrices, respectively
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000) # increase max_iter to 1000 (or another value if needed)

#model = LogisticRegression()

accuracy_scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    
print(f"Average accuracy: {sum(accuracy_scores) / len(accuracy_scores):.3f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# assuming X_scaled and y are the feature and target matrices, respectively
skf = StratifiedKFold(n_splits=12, shuffle=True, random_state=92)

model = LogisticRegression()

accuracy_scores = []

for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    
print(f"Average accuracy: {sum(accuracy_scores) / len(accuracy_scores):.3f}")
