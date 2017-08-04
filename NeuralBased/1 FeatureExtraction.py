# -*- coding: utf-8 -*-

import pandas as pd
import time
import string
import numpy as np
import datetime
import math

# define my stopwords list
sw_oj = open("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\stopwords from google history and MySQL.txt", 'r')
sw_oj = list(sw_oj)
MySW = ''
for i in range(0, len(sw_oj)):
    MySW += sw_oj[i]

MySW = MySW.replace('\n', ' ')
for punctuations in string.punctuation:
    MySW = MySW.replace(punctuations, ' ')

MySW = list(set(MySW.split()))
MySW += ['nan']


# Part 1
# Original Data (downloaded on Oct. 26, 2016 ) Initialization
od1 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats3a_securev1.csv", skiprows=[0, 39790, 42542, 42543])
od2 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats3b_securev1.csv", skiprows=[0, 188183, 188184])
od3 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats3c_securev1.csv", skiprows=[0, 235631, 235632])
od4 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats3d_securev1.csv", skiprows=[0, 421097, 421098])
od5 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats_securev1_2016Q1.csv", skiprows=[0, 133891, 133892])
od6 = pd.read_csv("D:\\Manuscript 5\\1 Original Data (download on 2016.10.26)\\LoanStats_securev1_2016Q2.csv", skiprows=[0, 97858, 97859])
original_data = od1.append([od2, od3, od4, od5, od6], ignore_index=True)
original_data.to_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\original_data.csv", index=False)

# Part 2
# Roughly Screening
clean = pd.read_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\original_data.csv")
# drop loans which haven't terminated yet
clean = clean[clean.loan_status == 'Fully Paid'].append(clean[clean.loan_status == 'Charged Off'], ignore_index=True)
# drop features which have only one value
features_to_drop = [clean.ix[:, i].name for i in range(0, clean.shape[1]) if len(clean.ix[:, i].unique()) == 1]
clean.drop(features_to_drop, axis=1, inplace=True)
clean.dropna(axis=1, how='all', inplace=True)
# drop features which have more than 20% NAs except desc (important textual info.)
desc = clean.desc
clean.dropna(axis=1, thresh=0.8*len(clean), inplace=True)
clean = clean.join(desc)

# Part 3
# Labels Generating
# Label 1 (1 for default)
clean.loan_status.replace('Fully Paid', 0, inplace=True)
clean.loan_status.replace('Charged Off', 1, inplace=True)
# Label 2 (loan's lifetime)
# term
clean.dropna(subset=['term'], inplace=True)
clean.term.replace(' 36 months', 36, inplace=True)
clean.term.replace(' 60 months', 60, inplace=True)
# issue date
clean.dropna(subset=['issue_d'], inplace=True)
issue_date = list(clean.issue_d.unique())
issue_timestamp = [int(time.mktime(time.strptime(date, '%b-%Y'))) for date in issue_date]  # unit: second
clean.issue_d.replace(issue_date, issue_timestamp, inplace=True)
# last payment date
clean.dropna(subset=['last_pymnt_d'], inplace=True)
last_payment_date = list(clean.last_pymnt_d.unique())
last_payment_timestamp = [int(time.mktime(time.strptime(date, '%b-%Y'))) for date in last_payment_date]  # unit: second
clean.last_pymnt_d.replace(last_payment_date, last_payment_timestamp, inplace=True)
# lifetime
clean.reset_index(drop=True, inplace=True)
py_times_rt = pd.Series(index=range(0, len(clean))).rename('py_times_rt')
py_times_co = pd.Series((clean.last_pymnt_d - clean.issue_d) // (60 * 60 * 24 * 30)).rename('py_times_co')
clean = clean.join(py_times_rt)
clean = clean.join(py_times_co)
fp = clean[clean.loan_status == 0]
fp['py_times_rt'] = fp['term']
co = clean[clean.loan_status == 1]
co['py_times_rt'] = co['py_times_co']
clean = fp.append(co, ignore_index=True)
clean.drop('py_times_co', axis=1, inplace=True)
clean.py_times_rt = clean.py_times_rt/clean.term
# others
clean = clean[clean.py_times_rt <= 1]
clean.reset_index(drop=True, inplace=True)
clean.issue_d.replace(issue_timestamp, issue_date, inplace=True)
clean.term.replace(36, '36 months', inplace=True)
clean.term.replace(60, '60 months', inplace=True)

# Part 4
# One-by-one Feature Processing
# 4.1 drop features obtained in future
clean.drop(['collection_recovery_fee', 'last_credit_pull_d', 'last_fico_range_high'], axis=1, inplace=True)
clean.drop(['last_fico_range_low', 'last_pymnt_amnt', 'last_pymnt_d'], axis=1, inplace=True)
clean.drop(['recoveries', 'pymnt_plan'], axis=1, inplace=True)
clean.drop(['total_pymnt', 'total_pymnt_inv', 'total_rec_int'], axis=1, inplace=True)
clean.drop(['total_rec_late_fee', 'total_rec_prncp'], axis=1, inplace=True)
# 4.2 drop useless features
clean.drop(['id', 'member_id', 'url', 'title'], axis=1, inplace=True)
# 4.3 term: expand term into dummies
clean = clean.join(pd.get_dummies(clean['term'], prefix='term').astype(int))
clean.drop(['term'], axis=1, inplace=True)
# 4.4 int_rate: strip out '%' and convert it into float
clean.int_rate = pd.Series(clean.int_rate).str.replace('%', '').astype(float)
# 4.5 grade and sub_grade: drop grade and create dummies for sub_grade
clean = clean.join(pd.get_dummies(clean['sub_grade'], prefix='grade').astype(int))
clean.drop(['grade', 'sub_grade'], axis=1, inplace=True)
# 4.6 emp_title: expand emp_title to dummies
clean.emp_title.fillna('NTAVLB', inplace=True)
# 4.6.1 delete punctuations
for punctuations in string.punctuation:
    clean.emp_title = clean.emp_title.str.replace(punctuations, ' ')

clean.emp_title = clean.emp_title.str.lower()
clean.emp_title = clean.emp_title.str.strip()
clean.emp_title = clean.emp_title.astype(str)
# 4.6.2 count words and delete stopwords
word_counts = pd.Series(' '.join(clean.emp_title).split()).value_counts()
for word in MySW:
    if word in word_counts:
        word_counts = word_counts.drop(word)
    else:
        continue


# 4.6.3 reclassify words
# management
mngt = ['manager', 'mgr', 'manger', 'director']
mngt += ['supervisor', 'president', 'executive']
mngt += ['vp', 'bod', 'ceo', 'cfo', 'coo', 'cio','cco', 'cso', 'cto']
# education
edu = ['teacher', 'professor', 'postdoctoral', 'scientist', 'educator', 'education']
edu += ['student', 'undergraduate', 'master', 'doctor', 'dr']
edu += ['school', 'university', 'universities', 'univercity', 'college', 'academy', 'academies']
# health care
medical = ['medical', 'medicine', 'nurse', 'nurses', 'nursing', 'rn', 'lpn', 'dentist', 'dental']
medical += ['pharmacy', 'pharmacies', 'pharmacist', 'paramedic', 'healthcare']
medical += ['clinical', 'clinic', 'physician', 'surgical', 'surgery']
medical += ['therapist', 'hygienist', 'hospital', 'health', 'care', 'caring']
# finance
fin = ['bank', 'financ', 'account', 'audit', 'securit', 'tax', 'broker', 'insurance', 'controller', 'underwrit']
fin += ['morgan', 'citi', 'merrill', 'goldman']
# company
corp = ['llc', 'compan', 'corporation', 'corp', 'llp', 'enterprise']
# law
law = ['attorney', 'legal', 'claim', 'law', 'court']
# admin
admin = ['admin']
# tech
tech = ['engineer', 'software', 'technology', 'technologies', 'programmer', 'computer', 'isd', 'tech', 'web', 'pc']
tech += ['ibm', 'huawei', 'cisco', 'apple', 'lenovo', 'facebook', 'twitter', 'google', 'ebay', 'amazon']
tech += ['htc', 'samsung', 'dell', 'microsoft', 'adobe', 'nvidia', 'amd', 'intel']
# car
car = ['ford', 'automotive', 'toyota', 'bmw', 'jeep', 'tesla', 'mercedes', 'benz', 'Volkswagen']
car += ['kia', 'mazda', 'audi', 'honda', 'hyundai', 'buick', 'chevrolet', 'nissan', 'land rover']
car += ['porsche', 'skoda', 'lexus', 'mitsubishi', 'volvo', 'cadillac', 'rolls royce', 'ferrari']
car += ['jaguar', 'infiniti', 'bentley', 'hummer', 'chrysler', 'lincoln', 'dodge']
# airplane
air = ['aircraft', 'lockheed', 'boeing', 'aviation']
# consult
consult = ['consult', 'advisor', 'counsel', 'adviser']
# express
express = ['postal', 'usps', 'fedex', 'transport', 'transpot', 'shipping', 'deliver', 'traffic', 'freight']
# power
power = ['electr', 'energy',  'power', 'resource', 'gas', 'petroleum', 'oil']
# govt
govt = ['officer', 'police', 'sheriff', 'inspector', 'superintendent', 'government', 'patrol']
govt += ['army', 'sergeant', 'navy', 'veterans', 'usaf', 'military', 'firefighter']
# skilled
skilled = ['technician', 'operator', 'mechanic', 'technologist', 'machinist', 'mechanical']
skilled += ['assembler', 'journeyman', 'driver', 'worker', 'welder']
# service
service = ['service', 'foreman', 'server', 'attendant', 'bartender', 'receptionist']
# 4.6.4 generate dummies
kws = [mngt, edu, medical, fin, corp, law, admin, tech, car, air, consult, express, power, govt, skilled, service]
for kw in kws:
    temp = pd.DataFrame(np.zeros(shape=(1,len(clean))).T)[0].rename('emp_title_'+kw[0])
    for i in range(0, len(kw)):
        temp += clean.emp_title.str.count(kw[i])
    clean = clean.join(temp)


clean.emp_title[clean.emp_title != 'ntavlb'] = 1
clean.emp_title[clean.emp_title == 'ntavlb'] = 0
clean.emp_title = clean.emp_title.astype(int)
# 4.7 emp_length: convert into a numerical variable
clean.emp_length.replace('n/a', np.nan, inplace=True)
clean.emp_length.fillna(value=0, inplace=True)
clean['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
clean['emp_length'] = clean['emp_length'].astype(int)
# 4.8 home_ownership: create dummies for it
clean.home_ownership.replace('ANY', 'NONE', inplace=True)
clean.home_ownership.replace('NONE', 'OTHER', inplace=True)
clean = clean.join(pd.get_dummies(clean['home_ownership'], prefix='home_ownership').astype(int))
clean.drop('home_ownership', axis=1, inplace=True)
# 4.9 verification_status: create dummies for it
clean = clean.join(pd.get_dummies(clean['verification_status'], prefix='verification_status').astype(int))
clean.drop('verification_status', axis=1, inplace=True)
# 4.10 issue_d: expand by year and month
# 4.10.1 YEAR
issue_d_2007 = pd.Series(clean.issue_d.str.contains('2007')).rename('issue_d_2007').astype(int)
clean = clean.join(issue_d_2007)
issue_d_2008 = pd.Series(clean.issue_d.str.contains('2008')).rename('issue_d_2008').astype(int)
clean = clean.join(issue_d_2008)
issue_d_2009 = pd.Series(clean.issue_d.str.contains('2009')).rename('issue_d_2009').astype(int)
clean = clean.join(issue_d_2009)
issue_d_2010 = pd.Series(clean.issue_d.str.contains('2010')).rename('issue_d_2010').astype(int)
clean = clean.join(issue_d_2010)
issue_d_2011 = pd.Series(clean.issue_d.str.contains('2011')).rename('issue_d_2011').astype(int)
clean = clean.join(issue_d_2011)
issue_d_2012 = pd.Series(clean.issue_d.str.contains('2012')).rename('issue_d_2012').astype(int)
clean = clean.join(issue_d_2012)
issue_d_2013 = pd.Series(clean.issue_d.str.contains('2013')).rename('issue_d_2013').astype(int)
clean = clean.join(issue_d_2013)
issue_d_2014 = pd.Series(clean.issue_d.str.contains('2014')).rename('issue_d_2014').astype(int)
clean = clean.join(issue_d_2014)
issue_d_2015 = pd.Series(clean.issue_d.str.contains('2015')).rename('issue_d_2015').astype(int)
clean = clean.join(issue_d_2015)
issue_d_2016 = pd.Series(clean.issue_d.str.contains('2016')).rename('issue_d_2016').astype(int)
clean = clean.join(issue_d_2016)
# 4.10.2 MONTH
issue_d_Jan = pd.Series(clean.issue_d.str.contains('Jan')).rename('issue_d_Jan').astype(int)
clean = clean.join(issue_d_Jan)
issue_d_Feb = pd.Series(clean.issue_d.str.contains('Feb')).rename('issue_d_Feb').astype(int)
clean = clean.join(issue_d_Feb)
issue_d_Mar = pd.Series(clean.issue_d.str.contains('Mar')).rename('issue_d_Mar').astype(int)
clean = clean.join(issue_d_Mar)
issue_d_Apr = pd.Series(clean.issue_d.str.contains('Apr')).rename('issue_d_Apr').astype(int)
clean = clean.join(issue_d_Apr)
issue_d_May = pd.Series(clean.issue_d.str.contains('May')).rename('issue_d_May').astype(int)
clean = clean.join(issue_d_May)
issue_d_Jun = pd.Series(clean.issue_d.str.contains('Jun')).rename('issue_d_Jun').astype(int)
clean = clean.join(issue_d_Jun)
issue_d_Jul = pd.Series(clean.issue_d.str.contains('Jul')).rename('issue_d_Jul').astype(int)
clean = clean.join(issue_d_Jul)
issue_d_Aug = pd.Series(clean.issue_d.str.contains('Aug')).rename('issue_d_Aug').astype(int)
clean = clean.join(issue_d_Aug)
issue_d_Sep = pd.Series(clean.issue_d.str.contains('Sep')).rename('issue_d_Sep').astype(int)
clean = clean.join(issue_d_Sep)
issue_d_Oct = pd.Series(clean.issue_d.str.contains('Oct')).rename('issue_d_Oct').astype(int)
clean = clean.join(issue_d_Oct)
issue_d_Nov = pd.Series(clean.issue_d.str.contains('Nov')).rename('issue_d_Nov').astype(int)
clean = clean.join(issue_d_Nov)
issue_d_Dec = pd.Series(clean.issue_d.str.contains('Dec')).rename('issue_d_Dec').astype(int)
clean = clean.join(issue_d_Dec)
# 4.11 purpose: create dummies for it
clean = clean.join(pd.get_dummies(clean['purpose'], prefix='purpose').astype(int))
clean.drop('purpose', axis=1, inplace=True)
# 4.12 zip_code and addr_state: create dummies for addr_state and drop zip code
clean = clean.join(pd.get_dummies(clean['addr_state'], prefix='addr_state').astype(int))
clean.drop(['zip_code', 'addr_state'], axis=1, inplace=True)
# 4.13 earliest_cr_line: duration from issue_d to earliest_cr_line
# 4.13.1 issue_d
issue_date = list(clean.issue_d.unique())
issue_timestamp = [int(time.mktime(time.strptime(date, '%b-%Y'))) for date in issue_date]  # unit: second
clean.issue_d.replace(issue_date, issue_timestamp, inplace=True)
# 4.13.2 earliest_cr_line
earliest_cr_line = list(clean.earliest_cr_line.unique())
epoch = datetime.datetime.strptime('1970-1-1', '%Y-%m-%d')
earliest_cr_line_timestamp = [((datetime.datetime.strptime(date, '%b-%Y') - epoch).days*24*3600+(datetime.datetime.strptime(date, '%b-%Y') - epoch).seconds) for date in earliest_cr_line]
clean.earliest_cr_line.replace(earliest_cr_line, earliest_cr_line_timestamp, inplace=True)
# 4.13.3 duration
clean.earliest_cr_line = clean.issue_d - clean.earliest_cr_line
# 4.14 revol_util: fill NaN; strip out '%' and convert it into float
clean.revol_util = pd.Series(clean.revol_util).str.replace('%', '').astype(float)
clean.revol_util.fillna(value=clean.revol_util.median(), inplace=True)
# 4.15 initial_list_status: create dummies for it
clean = clean.join(pd.get_dummies(clean['initial_list_status'], prefix='initial_list_status').astype(int))
clean.drop('initial_list_status', axis=1, inplace=True)
# 4.16 collections_12_mths_ex_med: fill NaN; convert from float to int
clean.collections_12_mths_ex_med.fillna(value=clean.collections_12_mths_ex_med.median(), inplace=True)
# 4.17 application_type: create dummies for it
clean = clean.join(pd.get_dummies(clean['application_type'], prefix='application_type').astype(int))
clean.drop('application_type', axis=1, inplace=True)
# 4.18 tot_coll_amt: fill NaN
clean.tot_coll_amt.fillna(value=clean.tot_coll_amt.median(), inplace=True)
# 4.19 tot_cur_bal: fill NaN
clean.tot_cur_bal.fillna(value=clean.tot_cur_bal.median(), inplace=True)
# 4.20 total_rev_hi_lim: fill NaN
clean.total_rev_hi_lim.fillna(value=clean.total_rev_hi_lim.median(), inplace=True)
# 4.21 acc_open_past_24mths: fill NaN
clean.acc_open_past_24mths.fillna(value=clean.acc_open_past_24mths.median(), inplace=True)
# 4.22 avg_cur_bal: fill NaN
clean.avg_cur_bal.fillna(value=clean.avg_cur_bal.median(), inplace=True)
# 4.23 bc_open_to_buy: fill NaN
clean.bc_open_to_buy.fillna(value=clean.bc_open_to_buy.median(), inplace=True)
# 4.24 bc_util: fill NaN
clean.bc_util.fillna(value=clean.bc_util.median(), inplace=True)
# 4.25 chargeoff_within_12_mths: fill NaN
clean.chargeoff_within_12_mths.fillna(value=clean.chargeoff_within_12_mths.median(), inplace=True)
# 4.26 delinq_amnt: fill NaN
clean.delinq_amnt.fillna(value=clean.delinq_amnt.median(), inplace=True)
# 4.27 mo_sin_old_il_acct: fill NaN
clean.mo_sin_old_il_acct.fillna(value=clean.mo_sin_old_il_acct.median(), inplace=True)
# 4.28 mo_sin_old_rev_tl_op: fill NaN
clean.mo_sin_old_rev_tl_op.fillna(value=clean.mo_sin_old_rev_tl_op.median(), inplace=True)
# 4.29 mo_sin_rcnt_rev_tl_op: fill NaN
clean.mo_sin_rcnt_rev_tl_op.fillna(value=clean.mo_sin_rcnt_rev_tl_op.median(), inplace=True)
# 4.30 mo_sin_rcnt_tl: fill NaN
clean.mo_sin_rcnt_tl.fillna(value=clean.mo_sin_rcnt_tl.median(), inplace=True)
# 4.31 mort_acc: fill NaN
clean.mort_acc.fillna(value=clean.mort_acc.median(), inplace=True)
# 4.32 mths_since_recent_bc: fill NaN
clean.mths_since_recent_bc.fillna(value=clean.mths_since_recent_bc.median(), inplace=True)
# 4.33 mths_since_recent_inq: fill NaN
clean.mths_since_recent_inq.fillna(value=clean.mths_since_recent_inq.median(), inplace=True)
# 4.34 num_accts_ever_120_pd: fill NaN
clean.num_accts_ever_120_pd.fillna(value=clean.num_accts_ever_120_pd.median(), inplace=True)
# 4.35 num_actv_bc_tl: fill NaN
clean.num_actv_bc_tl.fillna(value=clean.num_actv_bc_tl.median(), inplace=True)
# 4.36 num_actv_rev_tl: fill NaN
clean.num_actv_rev_tl.fillna(value=clean.num_actv_rev_tl.median(), inplace=True)
# 4.37 num_bc_sats: fill NaN
clean.num_bc_sats.fillna(value=clean.num_bc_sats.median(), inplace=True)
# 4.38 num_bc_tl: fill NaN
clean.num_bc_tl.fillna(value=clean.num_bc_tl.median(), inplace=True)
# 4.39 num_il_tl: fill NaN
clean.num_il_tl.fillna(value=clean.num_il_tl.median(), inplace=True)
# 4.40 num_op_rev_tl: fill NaN
clean.num_op_rev_tl.fillna(value=clean.num_op_rev_tl.median(), inplace=True)
# 4.41 num_rev_accts: fill NaN
clean.num_rev_accts.fillna(value=clean.num_rev_accts.median(), inplace=True)
# 4.42 num_rev_tl_bal_gt_0: fill NaN
clean.num_rev_tl_bal_gt_0.fillna(value=clean.num_rev_tl_bal_gt_0.median(), inplace=True)
# 4.43 num_sats: fill NaN
clean.num_sats.fillna(value=clean.num_sats.median(), inplace=True)
# 4.44 num_tl_120dpd_2m: fill NaN
clean.num_tl_120dpd_2m.fillna(value=clean.num_tl_120dpd_2m.median(), inplace=True)
# 4.45 num_tl_30dpd: fill NaN
clean.num_tl_30dpd.fillna(value=clean.num_tl_30dpd.median(), inplace=True)
# 4.46 num_tl_90g_dpd_24m: fill NaN
clean.num_tl_90g_dpd_24m.fillna(value=clean.num_tl_90g_dpd_24m.median(), inplace=True)
# 4.47 num_tl_op_past_12m: fill NaN
clean.num_tl_op_past_12m.fillna(value=clean.num_tl_op_past_12m.median(), inplace=True)
# 4.48 pct_tl_nvr_dlq: fill NaN
clean.pct_tl_nvr_dlq.fillna(value=clean.pct_tl_nvr_dlq.median(), inplace=True)
# 4.49 percent_bc_gt_75: fill NaN
clean.percent_bc_gt_75.fillna(value=clean.percent_bc_gt_75.median(), inplace=True)
# 4.50 pub_rec_bankruptcies: fill NaN
clean.pub_rec_bankruptcies.fillna(value=clean.pub_rec_bankruptcies.median(), inplace=True)
# 4.51 tax_liens: fill NaN
clean.tax_liens.fillna(value=clean.tax_liens.median(), inplace=True)
# 4.52 tot_hi_cred_lim: fill NaN
clean.tot_hi_cred_lim.fillna(value=clean.tot_hi_cred_lim.median(), inplace=True)
# 4.53 total_bal_ex_mort: fill NaN
clean.total_bal_ex_mort.fillna(value=clean.total_bal_ex_mort.median(), inplace=True)
# 4.54 total_bc_limit: fill NaN
clean.total_bc_limit.fillna(value=clean.total_bc_limit.median(), inplace=True)
# 4.55 total_il_high_credit_limit: fill NaN
clean.total_il_high_credit_limit.fillna(value=clean.total_il_high_credit_limit.median(), inplace=True)
# 4.56 desc: TF-IDF
# 4.56.1 cleansing
for punctuations in string.punctuation:
    clean.desc = clean.desc.str.replace(punctuations, ' ')

clean.desc = clean.desc.str.lower()
clean.desc = clean.desc.str.strip()
clean.desc = clean.desc.astype(str)
word_counts = pd.Series(' '.join(clean.desc).split()).value_counts()
# 4.56.2 delete stop words
for word in MySW:
    if word in word_counts:
        word_counts = word_counts.drop(word)
    else:
        continue


# 4.56.3 generate text length
desc_len = clean.desc.str.split().str.len().rename('desc_len')
clean = clean.join(desc_len)
clean.desc_len[clean.desc == 'nan'] = 0
# 4.56.4 construct my word list for TF-IDF
# 4.56.4.1 drop literal number rows and meaningless rows in word_counts
ltn = ['12', '13', '11', '10', '01', '07', '06', '09', '02', '14', '03', '08', '05', '04', '3']
ltn += ['2', '20', '15', '24', '18', '25', '17', '000', '16', '1', '19', '30', '00', '5', '26']
ltn += ['22', '29', '21', '23', '27', '28', '4', '6', '31', '99', '9', '7', '500', '100', '8']
ltn += ['36', '200', 'null', '300', '50', '400', '0', '600', '700', '800', '60', '1000', '2010']
ltn += ['2011', '40', '150', '2009', '2012', '2000', '250', '2008', '90']
mnl = ['br', 'amp', 'cc', 'mo', 'bit']
word_counts.drop(ltn, inplace=True)
word_counts.drop(mnl, inplace=True)
# 4.56.4.2 extract key words
kws = ['borrow', 'credit', 'loan', 'pay', 'paid', 'debt', 'card', 'interest', 'consolidat', 'year', 'rate', 'high']
kws += ['month', 'bill', 'job', 'money', 'low', 'time', 'plan', 'good', 'car', 'free', 'stable', 'work', 'expense']
kws += ['lend', 'current', 'business', 'club', 'compan', 'house', 'income', 'fund', 'late', 'sav', 'balance']
kws += ['amount', 'purchase', 'full', 'financ', 'great', 'personal', 'medic', 'family', 'account', 'budget']
kws += ['start', 'small', 'making', 'score', 'life', 'history', 'goal', 'apr ', 'cash', 'college', 'school']
kws += ['consider', 'mortgage', 'bank', 'reduc', 'minimum', 'excellent', 'buy', 'wedding', 'faster', 'close']
kws += ['future', 'employ', 'student', 'order', 'miss', 'exist', 'total', 'rent', 'repair', 'need', 'insurance']
kws += ['wife', 'quot', 'ago', 'hard', 'improv', 'extra', 'feel', 'responsible', 'purpose', 'term', 'rid ']
kws += ['long', 'mov', 'fix', 'eas', 'opportunit', 'question', 'position', 'invest', 'cover', 'revolv']
kws += ['secur', 'couple', 'single', 'remain', 'addition', 'liv', 'lot', 'vehicle', 'hop', 'eliminat', 'kitchen']
kws += ['left', 'thing', 'cost', 'forward', 'help', 'property', 'steady', 'soon', 'outstanding', 'track']
kws += ['large', 'care', 'auto', ' place', 'day', 'earl', 'major', 'half', 'increas', 'process', 'multiple']
kws += ['tax', 'add ', 'continue', 'set', 'finish', 'final', 'roof', 'request', 'quick', 'love', 'expect']
kws += ['career', 'replace', 'rest', 'people', 'cut', 'advance', 'pool', 'short', 'debit', 'emergency', 'capital']
kws += ['situation', 'support']
kws = list(set(kws))
# 4.56.4.3 counting
for kw in kws:
    temp = clean.desc.str.count(kw).rename('desc_' + kw.strip())
    clean = clean.join(temp)

# 4.56.5 compute TF
has_text = clean[clean.desc_len != 0]
no_text = clean[clean.desc_len == 0]
for kw in kws:
    temp = (has_text['desc_'+kw.strip()] / has_text['desc_len']).rename('tf_'+kw.strip())
    has_text = has_text.join(temp)

for kw in kws:
    temp = no_text['desc_len'].rename('tf_' + kw.strip())
    no_text = no_text.join(temp)

clean = has_text.append(no_text, ignore_index=True)
clean.reset_index(drop=True, inplace=True)
# 4.56.6 compute TF-IDF
len_nan_desc = len(clean[clean.desc == 'nan'])
for kw in kws:
    idf = math.log(len_nan_desc/((clean['desc_' + kw.strip()] != 0).sum()))
    temp = (idf * clean['tf_'+kw.strip()]).rename('tfidf_'+kw.strip())
    clean = clean.join(temp)

# 4.56.7 drop intermediate variables
desc_x = ['desc_'+kw.strip() for kw in kws]
tf_x = ['tf_'+kw.strip() for kw in kws]
clean.drop(desc_x, axis=1, inplace=True)
clean.drop(tf_x, axis=1, inplace=True)
# 4.56.8 generate dummy for nan
clean.desc[clean.desc != 'nan'] = 1
clean.desc[clean.desc == 'nan'] = 0
clean.desc = clean.desc.astype(int)
# 4.57 save data
clean.to_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\almost_cleaned_data.csv", index=False)

# Part 5
# Others
clean.pd.read_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\almost_cleaned_data.csv")
# 5.1 normalization
# 5.1.1 record some important values
clean.funded_amnt.mean()  # 13916.814225615919
clean.funded_amnt.std()  # 8255.782932371745
clean.installment.mean()  # 427.55053946249535
clean.installment.std()  # 248.0413326419988
# 5.1.2 normalization
clean_label1 = clean.loan_status
clean_label2 = clean.py_times_rt
clean.drop(['loan_status', 'py_times_rt'], axis=1, inplace=True)
clean = (clean-clean.mean())/clean.std()
# 5.2 re-order the columns
clean = clean.join(clean_label1)
clean = clean.join(clean_label2)
label1 = clean['loan_status']
clean.drop(labels=['loan_status'], axis=1, inplace=True)
clean.insert(0, 'loan_status', label1)
label2 = clean['py_times_rt']
clean.drop(labels=['py_times_rt'], axis=1, inplace=True)
clean.insert(0, 'py_times_rt', label2)
# 5.3 shuffle and save the cleaned data
np.random.seed(666); clean = clean.reindex(np.random.permutation(clean.index))
clean.reset_index(drop=True, inplace=True)
clean.to_csv("D:\\Manuscript 5\\3 Result\\1 FeatureExtraction\\cleaned_data.csv", index=False)




