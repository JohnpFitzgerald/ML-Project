# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:12:29 2023
@author: Jfitz
"""
#
# The following code (when run on the 24 hour data )will produce the 
# baseline results of the classifiers as described in Journal 2:
#          Logistic Regression	46% 
#          Random Forest	    63%
#          XG Boost	            63%
#          Light GBM	        64%
#          Decision tree	    54%
#          Keras Neural Network	66%
#
# 4 hourly features:
#
#          Logistic Regression	43% 
#          Random Forest	    53%
#          XG Boost	            53%
#          Light GBM	        54%
#          Decision tree	    48%
#          Keras Neural Network	55%             
#
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import seaborn as sns
color = sns.color_palette()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


file = '24HrFeatures.csv'
#file = '4HrFeatures.csv'
df = pd.read_csv(file)


#Prepare the data for modeling
X = df.copy().drop(['id','class','date','Category','counter','patientID'],axis=1)
y = df['class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the continuous variables
scaler = StandardScaler()
X_train[['f.mean', 'f.sd', 'f.propZeros']] = scaler.fit_transform(X_train[['f.mean', 'f.sd', 'f.propZeros']])
X_test[['f.mean', 'f.sd', 'f.propZeros']] = scaler.transform(X_test[['f.mean', 'f.sd', 'f.propZeros']])

#Create the models:

    
#----------------------Logistic Regression


logReg = LogisticRegression()

model = logReg

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1,2])

# Trainingset 10-fold cross validation
k_fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=2018)

# Fit the logistic regression model on the training data

model.fit(X_train, y_train)

# Evaluate the model's performance on the testing data
y_pred = model.predict(X_test)
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
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print("XGB Accuracy score:", acc_score)
print("XGB Confusion matrix:\n", conf_mat)
class_report = classification_report(y_test, y_pred)
print(f"XGB Classification Report:\n{class_report}")
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.barh(feat_imp.index, feat_imp.values)
plt.title("XGB Feature importances")
plt.show()


#-----------------------Light GBM


# Define the model
lgb_model = lgb.LGBMClassifier()

lgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lgb_model.predict(X_test)

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
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)

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


#----------------KERAS


# Step 2: Define the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 3: Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(" Neural Network Test accuracy:", test_acc)

