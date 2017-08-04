# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# method 1: Rating Based

train = pd.read_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\train_set.csv")
train = train[['py_times_rt', 'loan_status',
               'grade_A1', 'grade_A2', 'grade_A3', 'grade_A4', 'grade_A5',
               'grade_B1', 'grade_B2', 'grade_B3', 'grade_B4', 'grade_B5',
               'grade_C1', 'grade_C2', 'grade_C3', 'grade_C4', 'grade_C5',
               'grade_D1', 'grade_D2', 'grade_D3', 'grade_D4', 'grade_D5',
               'grade_E1', 'grade_E2', 'grade_E3', 'grade_E4', 'grade_E5',
               'grade_F1', 'grade_F2', 'grade_F3', 'grade_F4', 'grade_F5',
               'grade_G1', 'grade_G2', 'grade_G3', 'grade_G4', 'grade_G5']]
train.to_csv("D:\\Manuscript 6 for e-commerce workshop\\2 data\\train_for_rbm_comp.csv", index=False)








