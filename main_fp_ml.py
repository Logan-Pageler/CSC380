# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ipdb

from sklearn import tree

import mylib
import importlib
importlib.reload(mylib)
from mylib import *

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.svm import SVC


opt = SimpleNamespace()
opt.prefix = "./data/"
opt.pdata_name="pdata_v01.pkl"
printExpr('opt')

print("Loading data... ", end='')
tic()
df_efeatures = pd.read_csv(opt.prefix+"features.csv")
df_pnotes    = pd.read_csv(opt.prefix+"patient_notes.csv")
df_train     = pd.read_csv(opt.prefix+"train.csv")
df_test      = pd.read_csv(opt.prefix+"test.csv") 
dataw = DataWrapper(df_efeatures, df_pnotes, df_train, df_test)
print("Done (%.2fs)"% toc())

n_efeatures_list = [len(df_efeatures[df_efeatures.case_num == i]) for i in range(10)]
printExpr("n_efeatures_list")

# FIXME choose the processed data
out = LoadPickle('pdata_v01_120_numcom.pkl')
pdata = out.pdata
dataopt = out.dataopt
#opt = out.opt
ra.seed(29)

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=ra)

n_cases = 10
all_tp = 0
all_fp = 0
all_fn = 0
tps = np.zeros(n_cases)
fps = np.zeros(n_cases)
fns = np.zeros(n_cases)
for (case_num, mypdata) in enumerate(pdata):
    print('#--- case_num = %5d' % case_num)

    my_efeatures = np.unique(df_efeatures[df_efeatures.case_num == case_num].feature_num)
    n_efeatures = len(my_efeatures)

    my_pn_nums = np.unique(mypdata.pn_num)
    pairs = [(a,b) for a,b in kf.split(my_pn_nums)]

    all_predY = mypdata.label.copy()
    all_predY[:] = np.nan

    for train_idxs, val_idxs in pairs:
        train = mypdata[mypdata.pn_num.isin(my_pn_nums[train_idxs])]
        val = mypdata[mypdata.pn_num.isin(my_pn_nums[val_idxs])]

        trainX = train.iloc[:,3:-1]
        trainY = train.iloc[:,-1]

        valX = val.iloc[:,3:-1]
        valY = val.iloc[:,-1]

        # FIXME you can choose your classifier here. 
        # clf = tree.DecisionTreeClassifier()
        clf = RandomForestClassifier(ccp_alpha=0.00001)
        clf = clf.fit(trainX, trainY)
        predY = clf.predict(valX)

        all_predY[val.index] = predY
    assert ~np.any(np.isnan(all_predY))
    all_predY = all_predY.astype(int)
        
    #--- compute tp, fp, fn
    Y = mypdata.label

    n_efeatures = n_efeatures_list[case_num]

    eval_mat = np.zeros((len(my_pn_nums), n_efeatures, 3))
    for (i_pn_num, pn_num) in enumerate(my_pn_nums):
        pn_history = df_pnotes[df_pnotes.pn_num == pn_num].pn_history.values[0]

        # extract ground truth for each feature
        true_mat = dataw.get_ground_truth(case_num, pn_num)

        #- need to extract the location, still...
        pred_mat = true_mat.copy()
        pred_mat[:,:] = False

        my_index = mypdata[mypdata.pn_num == pn_num].index
        for idx in my_index:
            pred = all_predY[idx]
            if (pred != -1):
                from_ = mypdata.loc[idx,'loc_from']
                to_ = mypdata.loc[idx,'loc_to']
                
                pred_mat[pred, from_:to_] = True

        for efeature in range(n_efeatures):
            tp = (true_mat[efeature,:] & pred_mat[efeature,:]).sum()
            fp = (~true_mat[efeature,:] & pred_mat[efeature,:]).sum() 
            fn = (true_mat[efeature,:] & ~pred_mat[efeature,:]).sum() 
            eval_mat[i_pn_num,efeature,:] = [tp, fp, fn]
            pass
        pass

    tp = eval_mat[:,:,0].sum()
    fp = eval_mat[:,:,1].sum()
    fn = eval_mat[:,:,2].sum()
    prec = tp / (tp+fp)
    recall = tp / (tp + fn)
    f1 = 2/(1/recall + 1/prec)
    printExpr('prec')
    printExpr('recall')
    printExpr('f1')

    tps[case_num] = tp
    fps[case_num] = fp
    fns[case_num] = fn
    
print('\n#--- altogether')
tp = tps.sum()
fp = fps.sum()
fn = fns.sum()
prec = tp / (tp+fp)
recall = tp / (tp + fn)
f1 = 2/(1/recall + 1/prec)
printExpr('prec')
printExpr('recall')
printExpr('f1')

