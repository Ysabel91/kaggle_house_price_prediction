# coding: utf-8
"""
"""

##==================== Package ====================##
import pandas as pd
import sklearn



##==================== File-Path (fp) ====================##
## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = "./data/train_f.csv"
fp_test_f = "./data/test_f.csv"

## lr-model
fp_lr_model = "./data/lr/lr_model"

## submission files
fp_sub = "./data/lr/LR_submission.csv"

df_train_f = pd.read_csv(fp_train_f)

##==================== LR training ====================##


train_X = df_train_f.drop('SalePrice', axis=1)
train_y = df_train_f['SalePrice']
print(train_X.shape)
print(train_y.shape)

# LR
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(train_X, train_y)  # fitting
y_pred = lr_model.predict(train_X)
print(y_pred.shape)
score = sklearn.metrics.mean_squared_error(train_y, y_pred)
print(score)


# ramdom forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_X, train_y)
y_pred = rf.predict(train_X)
score2 = sklearn.metrics.mean_squared_error(train_y, y_pred)
print(score2)

# GDBT
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(random_state=10)
gbdt.fit(train_X, train_y)
trainGBDT_y = gbdt.predict(train_X)
score3 = sklearn.metrics.mean_squared_error(train_y, y_pred)
print(score3)



print(' - PY131 - ')