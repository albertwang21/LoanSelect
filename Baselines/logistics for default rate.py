# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import linear_model

train = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\train_set.csv")
test = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\test_set.csv")

train = train[['loan_status', 'fico_range_low', 'fico_range_high', 'loan_amnt', 'inq_last_6mths', 'dti', 'acc_now_delinq', 'home_ownership_OWN', 'home_ownership_RENT', 'int_rate', 'term_36 months']]
test = test[['loan_status', 'fico_range_low', 'fico_range_high', 'loan_amnt', 'inq_last_6mths', 'dti', 'acc_now_delinq', 'home_ownership_OWN', 'home_ownership_RENT', 'int_rate', 'term_36 months']]

train_x = train[['fico_range_low', 'fico_range_high', 'loan_amnt', 'inq_last_6mths', 'dti', 'acc_now_delinq', 'home_ownership_OWN', 'home_ownership_RENT', 'int_rate', 'term_36 months']]
train_y = train['loan_status']
train_x = train_x.as_matrix()
train_y = train_y.as_matrix()

test_x = test[['fico_range_low', 'fico_range_high', 'loan_amnt', 'inq_last_6mths', 'dti', 'acc_now_delinq', 'home_ownership_OWN', 'home_ownership_RENT', 'int_rate', 'term_36 months']]
test_y = test['loan_status']
test_x = test_x.as_matrix()
test_y = test_y.as_matrix()

lg_dr = linear_model.LogisticRegression()
lg_dr.fit(train_x, train_y)

default_rate = lg_dr.predict_proba(test_x)
default_rate = default_rate[:,1]
np.savetxt('D:\\Manuscript 6 for e-commerce workshop\\3 logit for default rate\\default_risk.csv', default_rate, fmt='%.20f')


