# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from __future__ import division

n_of_pf = 1000
pf_size = 80

irr = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\irr_distribution.csv")
realized = irr.realized_irr.as_matrix().reshape([n_of_pf,pf_size])

VaR_deep_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\weights_0.95alpha.npy")
VaR_deep_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\weights_0.99alpha.npy")
ES_deep_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\ES_weights_0.95alpha.npy")
ES_deep_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\2 data\\our\\ES_weights_0.99alpha.npy")

VaR_logit_deep_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\5 our with logit\\weights_0.95alpha.npy")
VaR_logit_deep_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\5 our with logit\\weights_0.99alpha.npy")
ES_logit_deep_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\5 our with logit\\ES_weights_0.95alpha.npy")
ES_logit_deep_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\5 our with logit\\ES_weights_0.99alpha.npy")

VaR_logit_rb_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\6 logit and new cd\\weights_0.95alpha.npy")
VaR_logit_rb_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\6 logit and new cd\\weights_0.99alpha.npy")
ES_logit_rb_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\6 logit and new cd\\ES_weights_0.95alpha.npy")
ES_logit_rb_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\6 logit and new cd\\ES_weights_0.99alpha.npy")

icdm_VaR_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\claimed_weights_95.npy")
icdm_VaR_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\claimed_weights_99.npy")
icdm_ES_95 = np.load("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\claimed_ES_weights_95.npy")
icdm_ES_99 = np.load("D:\\Manuscript 6 for e-commerce workshop\\7 icdm 2014\\claimed_ES_weights_99.npy")

rd_weights = np.empty([n_of_pf,pf_size])
for i in range(0, n_of_pf):
    rdwt = np.array([np.random.random() for j in range(0, pf_size)])
    rd_weights[i] = rdwt/rdwt.sum()

eq_weights = np.array([1/pf_size for i in range(0,80000)]).reshape([n_of_pf,pf_size])


def realized_portfolio_returns(x):
    portfolio_returns = np.empty([n_of_pf, ])
    for i in range(0, n_of_pf):
        portfolio_returns[i] = np.dot(x[i], realized[i])
    return portfolio_returns


def perts(x):
    for i in [1,5,10,15,20]:
        print(np.percentile(realized_portfolio_returns(x),i))


# final plot

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})

x1 = realized_portfolio_returns(ES_deep_95)
x2 = realized_portfolio_returns(ES_logit_deep_95)
x3 = realized_portfolio_returns(ES_logit_rb_95)
x4 = realized_portfolio_returns(icdm_ES_95)
x5 = realized_portfolio_returns(eq_weights)

plt.figure()
plt.hist([x1,x2,x3,x4,x5],bins=200,histtype='bar',label=['ES_deep_95','ES_logit_deep_95','ES_logit_rb_95','icdm_ES_95','eq_weights'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.07, 0.02])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()



x1 = realized_portfolio_returns(ES_deep_99)
x2 = realized_portfolio_returns(ES_logit_deep_99)
x3 = realized_portfolio_returns(ES_logit_rb_99)
x4 = realized_portfolio_returns(icdm_ES_99)
x5 = realized_portfolio_returns(eq_weights)

plt.figure()
plt.hist([x1,x2,x3,x4,x5],bins=50,histtype='bar',label=['ES_deep_99','ES_logit_deep_99','ES_logit_rb_99','icdm_ES_99','eq_weights'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.07, 0.02])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()





import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})

x1 = realized_portfolio_returns(ES_deep_99)
x2 = realized_portfolio_returns(ES_logit_rb_99)
x3 = realized_portfolio_returns(icdm_ES_99)


plt.figure()
plt.hist([x1,x2],bins=100,histtype='bar',color=['black','white'],label=['ES_dp_99','E99_lg_rb'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.06, 0.02])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()



