# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from __future__ import division
import scipy.optimize

# Part 1
# Data Preparation
mc_irr = np.load('D:\\Manuscript 5\\3 Result\\3 Simulation\\mc_irr.npy')
irr = pd.read_csv("D:\\Manuscript 5\\3 Result\\3 Simulation\\irr_distribution.csv")

# Part 2
# Define Optimization Problem
confidence = 0.99
pf_size = 80
n_of_pf = 1000

def VaR_mc(x):
    '''objective function 1'''
    x = np.array([x[i] for i in range(0, pf_size)])
    VaR = np.dot(x, pf_mc_irr)
    VaR.sort()
    return -VaR[int(10000*(1-confidence))-1]

def ES_mc(x):
    '''objective function 2'''
    x = np.array([x[i] for i in range(0, pf_size)])
    ES = np.dot(x, pf_mc_irr)
    ES.sort()
    return -ES[0:int(10000*(1-confidence))-1].mean()

def cons_eq(x):
    '''equation constraint'''
    sum = 0
    x = np.array([x[i] for i in range(0, pf_size)])
    for i in range(0, pf_size):
        sum += x[i]
    return sum - 1

# Part 3
# 3.1 initialization
ig = np.array([1/pf_size for i in range(0, pf_size)])
rg = tuple(((0, 1) for i in range(0, pf_size)))
cons = ({'type': 'eq',
         'fun' : cons_eq})
# 3.2 loop solving
weights = np.empty([n_of_pf,pf_size])
for i in range(0, n_of_pf):
    pf_mc_irr = mc_irr[pf_size * i: pf_size * (i+1)]
    res = scipy.optimize.minimize(VaR_mc, x0=ig, method='SLSQP', bounds=rg, constraints=cons, options={'disp': True})
    weights[i] = res.x

np.save("D:\\Manuscript 5\\3 Result\\4 Optimization\\weights_0.99alpha", weights)

ES_weights = np.empty([n_of_pf,pf_size])
for i in range(0, n_of_pf):
    pf_mc_irr = mc_irr[pf_size * i: pf_size * (i+1)]
    res = scipy.optimize.minimize(ES_mc, x0=ig, method='SLSQP', bounds=rg, constraints=cons, options={'disp': True})
    ES_weights[i] = res.x

np.save("D:\\Manuscript 5\\3 Result\\4 Optimization\\ES_weights_0.99alpha", ES_weights)

# 3.3 Comparison
irr = pd.read_csv("D:\\Manuscript 5\\3 Result\\3 Simulation\\irr_distribution.csv")
weights = np.load("D:\\Manuscript 5\\3 Result\\4 Optimization\\weights_0.99alpha.npy")
ES_weights = np.load("D:\\Manuscript 5\\3 Result\\4 Optimization\\ES_weights_0.99alpha.npy")

rd_weights = np.empty([n_of_pf,pf_size])
for i in range(0, n_of_pf):
    rdwt = np.array([np.random.random() for j in range(0, pf_size)])
    rd_weights[i] = rdwt/rdwt.sum()

eq_weights = np.array([1/pf_size for i in range(0,80000)]).reshape([n_of_pf,pf_size])

realized = irr.realized_irr.as_matrix().reshape([n_of_pf,pf_size])

# realized_weights
realized_weights = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    realized_weights[i] = np.multiply(weights[i], realized[i]).sum()

# realized_ES_weights
realized_ES_weights = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    realized_ES_weights[i] = np.multiply(ES_weights[i], realized[i]).sum()

# realized_rd_weights
realized_rd_weights = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    realized_rd_weights[i] = np.multiply(rd_weights[i], realized[i]).sum()

# realized_eq_weights
realized_eq_weights = np.empty([n_of_pf,])
for i in range(0,n_of_pf):
    realized_eq_weights[i] = np.multiply(eq_weights[i], realized[i]).sum()

# plot
realized_weights.sort()
realized_ES_weights.sort()
realized_rd_weights.sort()
realized_eq_weights.sort()

a = 20
np.percentile(realized_weights, a)
np.percentile(realized_ES_weights, a)
np.percentile(realized_rd_weights, a)
np.percentile(realized_eq_weights, a)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 30})

plt.figure()
plt.hist(realized_weights, bins=1000, color='red', label='portfolio returns(optimized)')
plt.hist(realized_rd_weights, bins=1000, color='yellow', label='portfolio returns(random-weighted)')
plt.hist(realized_eq_weights, bins=1000, color='green', label='portfolio returns(equal-weighted)')
plt.xlim([-0.2, 0.03])
plt.ylim([0, 100])
plt.xlabel('monthly irr')
plt.ylabel('frequency')
plt.title('monthly irr distribution')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(range(0,1000), realized_weights, color='red', lw=2, label='ascending portfolio returns(optimized)')
plt.plot(range(0,1000), realized_ES_weights, color='black', lw=2, label='ascending portfolio returns(optimized)')
plt.plot(range(0,1000), realized_rd_weights, color='yellow', lw=2, label='ascending portfolio returns(random)')
plt.plot(range(0,1000), realized_eq_weights, color='green', lw=2, label='ascending portfolio returns(equal)')
plt.xlim([-50, 1100])
plt.xlabel('portfolio')
plt.ylabel('monthly irr')
plt.title('realized monthly irr')
plt.legend(loc="lower right")
plt.show()

def gp(x):
    y = np.empty([10,])
    for i in range(0,10):
        y[i] = x[100*i:100*(i+1)].mean()
    return y

VaR_W = gp(realized_weights)
ES_W = gp(realized_ES_weights)
Rd_W = gp(realized_rd_weights)
Eq_W = gp(realized_eq_weights)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 30})

n_groups = 10

# means_men = (20, 35, 30, 35, 27)
# means_women = (25, 32, 34, 20, 25)

fig, ax = plt.subplots()
bw = 0.35

index = np.arange(n_groups) * 5 * bw

opacity = 1
rects1 = plt.bar(index, Rd_W, bw,alpha=opacity, color='white',label='Rd_W',edgecolor='black', hatch="/")
rects2 = plt.bar(index + bw, Eq_W, bw,alpha=opacity,color='white',label='Eq_W', edgecolor='black', hatch="\\")
rects3 = plt.bar(index + 2*bw, VaR_W, bw,alpha=opacity,color='grey',label='VaR_W',edgecolor='black')
rects4 = plt.bar(index + 3*bw, ES_W, bw,alpha=opacity,color='black',label='ES_W',edgecolor='black')

plt.xlabel('Groups')
plt.ylabel('monthly IRR')
plt.xlim([-bw,50*bw])
plt.ylim([-0.080,0.020])

plt.plot([-bw, 50*bw], [0, 0], lw=2, color='black', linestyle='-')

plt.xticks(index + 2*bw, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'))
plt.yticks([-0.07,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01], ('-0.07','-0.06','-0.05','-0.04','-0.03','-0.02','-0.01','0.00','0.01'))

plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# final plot

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

plt.figure()
plt.hist([realized_weights,realized_rd_weights],bins=200,histtype='bar',color=['black','white'],label=['VaR_99_W','Rd_W'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.1, 0.03])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.hist([realized_weights,realized_eq_weights],bins=200,histtype='bar',color=['black','white'],label=['VaR_99_W','Eq_W'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.1, 0.03])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.hist([realized_ES_weights,realized_rd_weights],bins=50,histtype='bar',color=['black','white'],label=['ES_99_W','Rd_W'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.1, 0.03])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.hist([realized_ES_weights,realized_eq_weights],bins=50,histtype='bar',color=['black','white'],label=['ES_99_W','Eq_W'])
plt.xlabel('monthly IRR')
plt.ylabel('Frequency')
plt.xlim([-0.1, 0.03])
plt.title('monthly IRR distribution')
plt.legend(loc="upper left")
plt.show()















