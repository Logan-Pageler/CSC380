# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm
from types import SimpleNamespace

opt = SimpleNamespace()
opt.kaggle_submission = False
opt.use_bigram = True #- FIXME: you can switch to bigram 
if (opt.kaggle_submission):
    opt.prefix = "/kaggle/input/nbme-score-clinical-patient-notes/"
else:
    opt.prefix = "./data/"
    import ipdb
    import mylib
    import importlib
    importlib.reload(mylib)
    from mylib import *
printExpr('opt')

dataopt = SimpleNamespace()
dataopt.W = 5
dataopt.bigramcutoff = 7
printExpr('dataopt')

print("Loading data... ", end='')
tic()
df_efeatures = pd.read_csv(opt.prefix+"features.csv")
df_pnotes    = pd.read_csv(opt.prefix+"patient_notes.csv")
df_train     = pd.read_csv(opt.prefix+"train.csv")
df_test      = pd.read_csv(opt.prefix+"test.csv") 
dataw = DataWrapper(df_efeatures, df_pnotes, df_train, df_test)
print("Done (%.2fs)"% toc())

#--- compute the distribution of the number of words in 'annotation'
rows = df_train.annotation
len_ary = []
for row in rows:
    str_ary = eval(row)
    for a_str in str_ary:
        len_ary.append( len(a_str.split()) )

from collections import Counter
cc = Counter(len_ary);
for k in sorted(cc.keys()):
    print("%02d: %6d" % (k, cc[k]))

n_efeatures_list = dataw.get_n_efeatures_list()
printExpr("n_efeatures_list")

n_cases = 10

#--- thedata : n_case by n_substring by n_features
#- each row, case_num, pn_num, loc_from, loc_to, ...

g_char_vocab = list('abcdefghijklmnopqrstuvwxyz0123456789-"\'/@')


if (opt.use_bigram):
    #------ first scan
    #- the goal is to collect the bigrams and compute the occurrences of each bigram throughout. 
    print('\n#----- first scan\n')
    from collections import Counter
    import re

    g_counter = None
    for case_num in range(n_cases):
        print('#--- case_num = %5d' % case_num)
        my_efeatures = np.unique(df_efeatures[df_efeatures.case_num == case_num].feature_num)
        n_efeatures = len(my_efeatures)
        my_train = df_train[df_train.case_num == case_num]
        my_pn_nums = my_train.pn_num

        #--- patient notes that are annotated & case_num = case_num
        my_pnotes = df_pnotes[(df_pnotes.case_num == case_num) & (df_pnotes.pn_num.isin(my_pn_nums)) ]
        my_pnotes.reset_index(inplace=True)

        out_table = []
        for i_row in tqdm(range(len(my_pnotes))):
            row = my_pnotes.loc[i_row]
            pn_history = row.pn_history
            pn_num = row.pn_num
            # sub = 0
            # hist = ""
            # length = len(pn_history)
            # for i, c in enumerate(pn_history):
            #     if c.isdigit():
            #         if i+1 < length and pn_history[i+1].isspace():
            #             hist += pn_history[sub:i+1]
            #             sub=i+2
            
            # if(sub < length):
            #     hist += pn_history[sub:length]
            
            cnt = Counter(calc_char_bigrams(pn_history))
            if (g_counter is None):
                g_counter = cnt
            else:
                g_counter += cnt
            pass
        pass

    bb = [[k,v] for k,v in g_counter.items()]
    bb = sorted(bb, key=lambda x: x[1])[::-1]

    print('first 10 bigrams:')
    cnt = 0
    for k,v in bb:
        print("%-7s : %7d" % (repr(k),v))
        cnt += 1
        if (cnt >= 10):
            break

    bigram_vocab = []
    for (k,v) in bb:
        if (v >= dataopt.bigramcutoff):
            bigram_vocab.append( k )

    bigram_vocab = sorted(bigram_vocab) 

#----- second scan

if opt.use_bigram:
    bigram_vocab_inv = dict(zip(bigram_vocab, range(len(bigram_vocab))))
    bigram_vocab_set = set(bigram_vocab)
    print(len(bigram_vocab_set))
    feature_extractor = lambda pn_history, cur_loc_mat: myfeatures_v02(pn_history, cur_loc_mat, g_char_vocab, bigram_vocab, bigram_vocab_set, bigram_vocab_inv)
else:
    feature_extractor = lambda pn_history, cur_loc_mat: myfeatures(pn_history, cur_loc_mat, g_char_vocab)

mde = MicroDataExtractor(dataopt.W, g_char_vocab, feature_extractor)
pdata = []
for case_num in range(n_cases):
    print('#--- case_num = %5d' % case_num)
    my_efeatures = np.unique(df_efeatures[df_efeatures.case_num == case_num].feature_num)
    n_efeatures = len(my_efeatures)
    my_train = df_train[df_train.case_num == case_num]
    my_pn_nums = my_train.pn_num

    #--- patient notes that are annotated & case_num = case_num
    my_pnotes = df_pnotes[(df_pnotes.case_num == case_num) & (df_pnotes.pn_num.isin(my_pn_nums)) ]
    my_pnotes.reset_index(inplace=True)

    out_table = []
    for i_row in tqdm(range(len(my_pnotes))):
        row = my_pnotes.loc[i_row]
        pn_history = row.pn_history
        pn_num = row.pn_num

        # extract ground truth for each feature
        true_mat = dataw.get_ground_truth(case_num, pn_num)

        my_out_table, col_names = mde.extract_micro_data(pn_num, pn_history, true_mat)
        out_table += my_out_table
        pass
        
    pdata.append( pd.DataFrame(out_table, columns=col_names) )
    pass

out = SimpleNamespace()
out.pdata = pdata
out.dataopt = dataopt
out.char_vocab = g_char_vocab
if (opt.use_bigram):
    out.bigram_vocab = bigram_vocab

if (not opt.kaggle_submission):
    SavePickle("pdata_v01_RENAME.pkl", out) # c0 for case_num = 0



