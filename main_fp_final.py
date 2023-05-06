# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from types import SimpleNamespace

opt = SimpleNamespace()
opt.kaggle_submission = False
opt.use_bigram = False
opt.debug = False
if (opt.kaggle_submission):
    opt.prefix = "/kaggle/input/nbme-score-clinical-patient-notes/"
else:
    opt.prefix = "./"
    import ipdb
    import mylib
    import importlib
    importlib.reload(mylib)
    from mylib import *
printExpr('opt')

n_cases = 10

#--- load data
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

# #---------------------------
if (not opt.kaggle_submission):
    # FIXME: change the file
    out = LoadPickle('pdata_v01.pkl')

pdata = out.pdata
char_vocab = out.char_vocab
if (opt.use_bigram):
    bigram_vocab = out.bigram_vocab
dataopt = out.dataopt


ra.seed(39)

#---- let's train
clf_list = []
for (case_num, mypdata) in enumerate(pdata):
    print('#--- case_num = %5d' % case_num)

    trainX = mypdata.iloc[:,3:-1]
    trainY = mypdata.iloc[:,-1]

    # FIXME choose your classifier here
    clf = tree.DecisionTreeClassifier()
    if opt.debug:
        clf = clf.fit(trainX[:400],trainY[:400])
    else:
        clf = clf.fit(trainX,trainY)
    clf_list.append(clf)

#---- let's test
predictions = []

if opt.use_bigram:
    bigram_vocab_inv = dict(zip(bigram_vocab, range(len(bigram_vocab))))
    bigram_vocab_set = set(bigram_vocab)
    feature_extractor = lambda pn_history, cur_loc_mat: myfeatures_v02(pn_history, cur_loc_mat, char_vocab, bigram_vocab, bigram_vocab_set, bigram_vocab_inv)
else:
    feature_extractor = lambda pn_history, cur_loc_mat: myfeatures(pn_history, cur_loc_mat, char_vocab)
mde = MicroDataExtractor(dataopt.W, char_vocab, feature_extractor)

pn_num_ary = np.unique(df_test.pn_num)
answers = dict()
for (i,pn_num) in enumerate(pn_num_ary):
    pn_history = dataw.get_pn_history(pn_num)

    my_out_table, col_names = mde.extract_micro_data(pn_num, pn_history)

    test = pd.DataFrame(my_out_table, columns = col_names)
    case_num = dataw.get_case_num(pn_num)
    my_efeatures = dataw.get_efeatures()[case_num]
    my_n_efeatures = len(my_efeatures)
    my_clf = clf_list[case_num]

    testX = test.iloc[:,3:-1]

    #- predict
    predY = my_clf.predict(testX)
    pred_mat = np.zeros((my_n_efeatures,len(pn_history)), dtype=bool)
    pred_mat[:,:] = False

    #- make prediction for each phrase.
    for i in range(len(test)):
        pred = predY[i]
        if (pred != -1):
            from_ = test.iloc[i].loc_from
            to_   = test.iloc[i].loc_to
            pred_mat[pred, from_:to_] = True

    #- go from pred_mat to to loc_string
    pn_answers = dict()
    for i_ef, efeature_num in enumerate(my_efeatures):
        pn_answers[efeature_num] = get_loc_string(pred_mat[i_ef,:])

    answers[pn_num] = pn_answers

#--- let's write out the answer
predictions = []
for i in range(len(df_test)):
    row = df_test.iloc[i]
    pn_num = row.pn_num
    case_num = dataw.get_case_num(pn_num)
    predictions.append(answers[pn_num][row.feature_num])
    
output = pd.DataFrame({'id':df_test.id, 'location':predictions})
output.to_csv("submission.csv", index=False)




