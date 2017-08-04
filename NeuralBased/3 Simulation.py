# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from __future__ import division

# Part 1
# Data Initialization
test = pd.read_csv("D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\test_set.csv")
cd_lifetime_rt = np.loadtxt('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\cd_lt\\lifetime_default.csv')
cd_lifetime_rt = pd.Series(cd_lifetime_rt).rename('cd_lifetime_rt')
test = test.join(cd_lifetime_rt)
default_risk = np.loadtxt('D:\\Manuscript 5\\3 Result\\2 Paramecium 5.0\\default_risk\\default_risk.csv')
default_risk = pd.Series(default_risk).rename('default_risk')
test = test.join(default_risk)
term = test['term_36 months'].replace([-1.8138064252799999, 0.55132538318699997], [60, 36])
test = test.join(term.rename('cd_lifetime_non_dft').astype(int))
test.cd_lifetime_rt = (test.cd_lifetime_rt * term).round()
test = test.join(test.cd_lifetime_rt.rename('cd_lifetime_dft').astype(int))
test.funded_amnt = test.funded_amnt * 8255.782932371745 + 13916.814225615919
test.installment = test.installment * 248.0413326419988 + 427.55053946249535
py_times = (test.py_times_rt * test.cd_lifetime_non_dft).rename('py_times').astype(int)
test = test.join(py_times)
data = test[['cd_lifetime_dft', 'cd_lifetime_non_dft', 'funded_amnt', 'installment', 'py_times', 'default_risk']]

# Part 2
# Generate Distributions of loan returns
# 2.1 claim_irr
claim_irr = pd.Series(index=range(0, len(data))).rename('claim_irr')
data = data.join(claim_irr)
for i in range(0, len(data)):
    cf = [-data.funded_amnt[i]] + [data.installment[i]] * data.cd_lifetime_non_dft[i]
    data.claim_irr[i] = np.irr(cf)

# 2.2 default_irr
default_irr = pd.Series(index=range(0, len(data))).rename('default_irr')
data = data.join(default_irr)
for i in range(0, len(data)):
    cf = [-data.funded_amnt[i]] + [data.installment[i]] * data.cd_lifetime_dft[i]
    data.default_irr[i] = np.irr(cf)

# 2.3 expected_irr
expected_irr = pd.Series(index=range(0, len(data))).rename('expected_irr')
data = data.join(expected_irr)
data.expected_irr = data.claim_irr * (1- data.default_risk) + data.default_irr * data.default_risk
# 2.4 realized_irr
realized_irr = pd.Series(index=range(0, len(data))).rename('realized_irr')
data = data.join(realized_irr)
for i in range(0, len(data)):
    if data.py_times[i]==0:
        data.realized_irr[i] = -1
    else:
        cf = [-data.funded_amnt[i]] + [data.installment[i]] * data.py_times[i]
        data.realized_irr[i] = np.irr(cf)


data = data[['claim_irr', 'default_irr', 'default_risk', 'expected_irr', 'realized_irr']]
np.random.seed(111); data = data.reindex(np.random.permutation(data.index))
data.reset_index(drop=True, inplace=True)
data.to_csv("D:\\Manuscript 5\\3 Result\\3 Simulation\\irr_distribution.csv", index=False)

# Part 3
# 10000 Events Generating using Monte-Carlo Simulation
data = pd.read_csv("D:\\Manuscript 5\\3 Result\\3 Simulation\\irr_distribution.csv")
mc = np.empty([len(data), 10000])
for i in range(0, len(data)):
    p = data.default_risk[i]
    np.random.seed(i); temp = np.random.binomial(1, p, 10000)
    mc[i] = temp

np.save('D:\\Manuscript 5\\3 Result\\3 Simulation\\monte_carlo_events.npy', mc)

# Part 4
# Replace Events with Conditional IRR
mc_dft = np.load('D:\\Manuscript 5\\3 Result\\3 Simulation\\monte_carlo_events.npy')
mc_ndft = 1 - mc_dft
data = pd.read_csv("D:\\Manuscript 5\\3 Result\\3 Simulation\\irr_distribution.csv")
data = data.round(6)
claim_irr = np.empty([10000, 80000])
claim_temp = data.claim_irr.as_matrix()
for i in range(0,10000):
    claim_irr[i] = claim_temp

claim_irr = claim_irr.T

default_irr = np.empty([10000, 80000])
default_temp = data.default_irr.as_matrix()
for i in range(0,10000):
    default_irr[i] = default_temp

default_irr = default_irr.T

np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\default_irr.npy", default_irr)
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\claim_irr.npy", claim_irr)
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_dft.npy", mc_dft)
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_ndft.npy", mc_ndft)

# Memory Management
default_irr = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\default_irr.npy")
mc_dft = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_dft.npy")
default_irr = np.multiply(default_irr, mc_dft)
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_dft_irr.npy", default_irr)

claim_irr = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\claim_irr.npy")
mc_ndft = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_ndft.npy")
claim_irr = np.multiply(claim_irr, mc_ndft)
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_ndft_irr.npy", claim_irr)

mc_dft_irr = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_dft_irr.npy")
mc_ndft_irr = np.load("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_ndft_irr.npy")

mc_irr = mc_ndft_irr + mc_dft_irr
np.save("D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_irr.npy", mc_irr)


