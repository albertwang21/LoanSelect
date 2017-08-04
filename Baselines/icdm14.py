# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from __future__ import division
import scipy.optimize

# Part 1
# Data Initialization
test = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\test_set.csv")

default_risk = np.loadtxt('D:\\Manuscript 6 for e-commerce workshop\\3 logit for default rate\\default_risk.csv')
default_risk = pd.Series(default_risk).rename('default_risk')
test = test.join(default_risk)

term = test['term_36 months'].replace([-1.8138064252799999, 0.55132538318699997], [60, 36])
test = test.join(term.rename('cd_lifetime_non_dft').astype(int))

test.funded_amnt = test.funded_amnt * 8255.782932371745 + 13916.814225615919
test.installment = test.installment * 248.0413326419988 + 427.55053946249535
py_times = (test.py_times_rt * test.cd_lifetime_non_dft).rename('py_times').astype(int)
test = test.join(py_times)

data = test[['cd_lifetime_non_dft', 'funded_amnt', 'installment', 'py_times', 'default_risk']]

claim_irr = pd.Series(index=range(0, len(data))).rename('claim_irr')
data = data.join(claim_irr)
for i in range(0, len(data)):
    cf = [-data.funded_amnt[i]] + [data.installment[i]] * data.cd_lifetime_non_dft[i]
    data.claim_irr[i] = np.irr(cf)


realized_irr = pd.Series(index=range(0, len(data))).rename('realized_irr')
data = data.join(realized_irr)
for i in range(0, len(data)):
    if data.py_times[i]==0:
        data.realized_irr[i] = -1
    else:
        cf = [-data.funded_amnt[i]] + [data.installment[i]] * data.py_times[i]
        data.realized_irr[i] = np.irr(cf)


data = data[['claim_irr', 'realized_irr', 'default_risk']]
np.random.seed(111); data = data.reindex(np.random.permutation(data.index))
data.reset_index(drop=True, inplace=True)
data.to_csv("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\irr_distribution.csv", index=False)


irr = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\irr_distribution.csv")
weights_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\weights_0.95alpha.npy")
ES_weights_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\ES_weights_0.95alpha.npy")
weights_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\weights_0.99alpha.npy")
ES_weights_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\ES_weights_0.99alpha.npy")

pf_size = 80
n_of_pf = 1000

claimed = irr.claim_irr.as_matrix().reshape([n_of_pf,pf_size])

# claimed_weights_95
claimed_weights_95 = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    claimed_weights_95[i] = np.multiply(weights_95[i], claimed[i]).sum()

# claimed_ES_weights_95
claimed_ES_weights_95 = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    claimed_ES_weights_95[i] = np.multiply(ES_weights_95[i], claimed[i]).sum()

# claimed_weights_99
claimed_weights_99 = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    claimed_weights_99[i] = np.multiply(weights_99[i], claimed[i]).sum()

# claimed_ES_weights_99
claimed_ES_weights_99 = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    claimed_ES_weights_99[i] = np.multiply(ES_weights_99[i], claimed[i]).sum()


claimed_weights_95.sum()/1000
# 0.010143810798042415

claimed_ES_weights_95.sum()/1000
# 0.0087068530996656391

claimed_weights_99.sum()/1000
# 0.0094307420583124319

claimed_ES_weights_99.sum()/1000
# 0.0090951882197153645


# Part 2
# Optimization
irr = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\irr_distribution.csv")

claimed_weights_95 = 0.010143810798042415
claimed_ES_weights_95 = 0.0087068530996656391
claimed_weights_99 = 0.0094307420583124319
claimed_ES_weights_99 = 0.0090951882197153645

required_irr = claimed_ES_weights_99

pf_size = 80
n_of_pf = 1000

claim_irr = irr.claim_irr.as_matrix().reshape([n_of_pf,pf_size])
default_rate = irr.default_risk.as_matrix().reshape([n_of_pf,pf_size])


def Variance(x):
    '''objective function 1'''
    x = np.array([x[i] for i in range(0, pf_size)])
    Variance = np.multiply(x, dr)
    Variance = np.multiply(Variance, Variance)
    return Variance.sum()


def cons_eq1(x):
    '''equation constraint 1'''
    x = np.array([x[i] for i in range(0, pf_size)])
    return np.dot(x, ci) - required_irr


def cons_eq2(x):
    '''equation constraint 2'''
    sum = 0
    x = np.array([x[i] for i in range(0, pf_size)])
    for i in range(0, pf_size):
        sum += x[i]
    return sum - 1


ig = np.array([1/pf_size for i in range(0, pf_size)])
rg = tuple(((0, 1) for i in range(0, pf_size)))
cons = ({'type': 'eq', 'fun' : cons_eq1},
        {'type': 'eq', 'fun': cons_eq2})


weights = np.empty([n_of_pf,pf_size])
for i in range(0, n_of_pf):
    dr = default_rate[i]
    ci = claim_irr[i]
    res = scipy.optimize.minimize(Variance, x0=ig, method='SLSQP', bounds=rg, constraints=cons, options={'disp': True})
    weights[i] = res.x

np.save("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\claimed_ES_weights_99", weights)





