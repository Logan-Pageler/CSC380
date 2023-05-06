import numpy as np
import numpy.random as ra
import re
from collections import Counter
import traceback, sys, pickle, os
from types import SimpleNamespace

#import ipdb



################################################################################
# basic utility functions
################################################################################

def nans(*args):
    ary = np.zeros(*args)
    ary.fill(np.nan)
    return ary

def LoadPickle(fName):
    """ load a pickle file. Assumes that it has one dictionary object that points to
 many other variables."""
    if type(fName) == str:
        try:
            fp = open(fName, 'rb')
        except:
            print("Couldn't open %s" % (fName))
            traceback.print_exc(file=sys.stderr)
    else:
        fp = fName
    try:
        ind = pickle.load(fp)
        fp.close()
        return ind
    except:
        print("Couldn't read the pickle file", fName)
        traceback.print_exc(file=sys.stderr)

def SavePickle(filename, var, protocol=2):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(var, f, protocol=protocol)
        statinfo = os.stat(filename,)
        if statinfo:
            print("Wrote out", statinfo.st_size, "bytes to", \
                filename)
    except:
        print("Couldn't pickle the file", filename)
        traceback.print_exc(file=sys.stderr)


def printExpr(expr, bPretty=True):
    """ Print the local variables in the caller's frame."""
    from pprint import pprint
    import inspect
    frame = inspect.currentframe()
    try:
        loc = frame.f_back.f_locals
        glo = frame.f_back.f_globals
        print(expr, '= ', end=' ') 
        if (bPretty):
            pprint(eval(expr, glo, loc))
        else:
            print((eval(expr, glo, loc))); 
    finally:

        del frame

from datetime import datetime

def tic():
    """
    equivalent to Matlab's tic. It start measuring time.
    returns handle of the time start point.
    """
    global gStartTime
    gStartTime = datetime.utcnow()
    return gStartTime

def toc(prev=None):
    """
    get a timestamp in seconds. Time interval is from previous call of tic() to current call of toc().
    You can optionally specify the handle of the time ending point.
    """
    if prev==None: prev = gStartTime
    return (datetime.utcnow() - prev).total_seconds()

################################################################################
# extracting features
################################################################################
g_out_of_char_vocab = set()
def calc_char_unigram_features(s, vocab):
    global g_out_of_char_vocab
    paired = zip(vocab, np.arange(len(vocab)))
    vocab_to_idx = dict(paired)

    ss = s.strip().lower()
    res = [list(ss[ele.start():ele.end()]) for ele in re.finditer(r'[^ \t\n\r\f\v,.;:()]+', ss)]

    # flatten it out for unigram
    flattened = [y for x in res for y in x]
    cntr = Counter(flattened)
    vec = np.zeros(len(vocab))
    for k,v in cntr.items():
        if (k not in vocab_to_idx):
            g_out_of_char_vocab.add(k)
        else:
            vec[vocab_to_idx[k]] = v

    if (vec.sum() != 0.0):
        vec /= vec.sum()
    return vec

def calc_char_bigrams(text):
    ### TODO insert code here.
    text = re.sub(r"[!,\.:;?]", " ", text)

    words = text.split()

    bigrams=[]

    for word in words:
        bigrams.append(word[0])
        for i in range(0, len(word) -1):
            bigrams.append(word[i:i+2])
        bigrams.append(word[-1])
    
    return bigrams

def calc_char_bigram_features(s, vocab, vocab_set, vocab_inv):
    ss = ' ' + s.strip().lower() + ' '
    bigrams = list(filter(lambda x: x in vocab_set, calc_char_bigrams(ss)))

    cnt = Counter(bigrams)
    #vec = np.array( [cnt[k] for k in vocab] , dtype=float)
    vec = np.zeros(len(vocab))
    for (k,v) in cnt.items():
        vec[vocab_inv[k]] = v

    total = vec.sum()

    if (total != 0.0):
        vec /= total

    return vec

################################################################################
# for data processing
################################################################################

class DataWrapper:
    def __init__(self, df_features, df_pnotes, df_train, df_test):
        self.df_features = df_features
        self.df_pnotes = df_pnotes
        self.df_train = df_train
        self.df_test  = df_test
        self.efeatures = []
        self.n_efeatures_list = []
        for i_case in range(10):
            self.efeatures.append( np.unique(df_features[df_features.case_num == i_case].feature_num).tolist() )
            self.n_efeatures_list.append( len(self.efeatures[i_case]) )
        self.n_efeatures_list = np.array(self.n_efeatures_list)

        #- obtain pnotes length
        # go through pnotes with valid pn_num.
        self.pn_history_len = dict()
        for i in range(len(self.df_pnotes)):
            my = self.df_pnotes.iloc[i]
            self.pn_history_len[my.pn_num] = len(my.pn_history)

    def get_efeatures(self):
        return self.efeatures

    def get_n_efeatures_list(self):
        return self.n_efeatures_list

    def get_ground_truth(self, case_num, pn_num):
        """
        returns the ground truth in the form of logical matrix of size (n_efeatures, length of pn_history for pn_num)
        """
        n_efeatures = self.n_efeatures_list[case_num] 
        my_efeatures = self.efeatures[case_num]

        true_mat = np.zeros((n_efeatures, self.pn_history_len[pn_num]), dtype=bool)
        cond_pn_num = (self.df_train.pn_num == pn_num)

        for i_ef, efeature_num in enumerate(my_efeatures):
            loc = self.df_train[cond_pn_num \
                              & (self.df_train.feature_num == efeature_num)].location
            for (from_,to_) in location_str_to_ary(loc.tolist()[0]):
                true_mat[i_ef, from_:to_] = True

        return true_mat

    def get_case_num(self,pn_num):
        return self.df_pnotes[self.df_pnotes.pn_num == pn_num].case_num.tolist()[0]

    def get_pn_history(self, pn_num):
        return self.df_pnotes[self.df_pnotes.pn_num == pn_num].pn_history.tolist()[0]
    pass

def loc_is_in(loc, true_loc_mat):
    """
    true_loc is n by 2
    """
    ret_val = False
    for i in range(len(true_loc_mat)):
        if (loc[0] >= true_loc_mat[i][0] and loc[1] <= true_loc_mat[i][1]):
            ret_val = True
            break
    return ret_val

def get_word_loc_ary(text):
    """
    a list of tuple (start_index, end_index) indicating each word location; note we follow python's standard indexing, so the word starts from start_index but ends at end_index-1
    """
    return np.array([(ele.start(), ele.end()) for ele in re.finditer(r'[^ \t\n\r\f\v,.;:()]+', text.strip())])


def get_loc_string(pred_ary):
    """
        In [32]: get_loc_string(np.array([1,0,1,1,0,1],dtype=bool))
        Out[32]: '0 1;2 4;5 6'
    """    
    switch = False
    locations = []
    n = len(pred_ary)
    for i in range(-1,n):
        # v: value at i
        # vv: value at i+1
        if (i == -1):
            v = False # sentinel
        else:
            v = pred_ary[i]

        if (i == n - 1):
            vv = False # sentinel
        else:
            vv = pred_ary[i+1]

        if (np.logical_xor(v,vv)):
            locations.append(i+1)

    assert len(locations) % 2 == 0
    loc = np.array(locations)
    loc = loc.reshape(-1,2)
    return ';'.join(['%d %d' % (row[0], row[1])for row in loc])

def location_str_to_ary(loc):
    """
    In [102]: location_str_to_ary("['595 724']")
    Out[102]: [[595, 724]]

    In [100]: location_str_to_ary("['595 724', '652 661']")
    Out[100]: [[595, 724], [652, 661]]

    In [57]: location_str_to_ary("['595 724', '652 661; 665 670']")
    Out[57]: [[595, 724], [652, 661], [665, 670]]

    In [101]: location_str_to_ary("[]")
    Out[101]: []
    """    
    myloc = "', '".join(loc.split(';'))
    myloc = eval(myloc)
    # later, I may need to use binary array representation..
    #     tmp = [[int(j) for j in pair_str.split()] for pair_str in myloc]
    #     maxval = max([row[1] for row in tmp])
    # 
    #     bin_ary = np.zeros(maxval, dtype=bool)
    #     for intv in range(len(tmp)):
    #         bin_ary[intv[0]:intv[1]] = True
    # 
    #     bin_ary

    return [[int(j) for j in pair_str.split()] for pair_str in myloc]

################################################################################
# for feature extraction
################################################################################

def myfeatures(pn_history, cur_loc_mat, char_vocab):
    cur_loc_all = [cur_loc_mat[0,0], cur_loc_mat[-1,1]]
    
    fvec_names = []
    fvec = []

    fvec_names.append( 'n_chars' )
    fvec.append( np.sum([loc[1] - loc[0] for loc in cur_loc_mat]) )

    fvec_names.append( 'n_words' )
    fvec.append( cur_loc_mat.shape[0] )

    fvec_names += ['char_unigram_' + c for c in char_vocab]
    text = pn_history[cur_loc_all[0]:cur_loc_all[1]]
    fvec += calc_char_unigram_features(text, char_vocab).tolist()

    return fvec_names, fvec


def myfeatures_v02(pn_history, cur_loc_mat, char_vocab, bigram_vocab, bigram_vocab_set, bigram_vocab_inv):
    cur_loc_all = [cur_loc_mat[0,0], cur_loc_mat[-1,1]]
    
    fvec_names = []
    fvec = []

    fvec_names.append( 'n_chars' )
    fvec.append( np.sum([loc[1] - loc[0] for loc in cur_loc_mat]) )

    fvec_names.append( 'n_words' )
    fvec.append( cur_loc_mat.shape[0] )

    text = pn_history[cur_loc_all[0]:cur_loc_all[1]]

    fvec_names += ['char_unigram_' + c for c in char_vocab]
    fvec += calc_char_unigram_features(text, char_vocab).tolist()

    fvec_names += ['char_bigram_' + repr(c) for c in bigram_vocab]
    fvec += calc_char_bigram_features(text, bigram_vocab, bigram_vocab_set, bigram_vocab_inv).tolist()

    return fvec_names, fvec

class MicroDataExtractor:
    def __init__(self, W, char_vocab, feature_extractor):
        """
        feature_extractor must be a function that takes (pn_history, cur_loc_mat) and output the feature vector representation
        """
        self.W = W
        self.char_vocab = char_vocab
        assert feature_extractor is not None
        self.feature_extractor = feature_extractor

        pass

    def extract_micro_data(self, pn_num, pn_history, true_mat=None):
        """
           true_mat: np.array with size (n_efeatures, length of pn_history)
                     if None, we do not extract the label (this is true for test data) 
        """
        out_table = []
        loc_ary = get_word_loc_ary(pn_history)
        for i_loc in range(loc_ary.shape[0]):
            # FIXME could use the generator pattern
            # for w in range 
            for w in range(1, 1+self.W):
                if (i_loc+w > loc_ary.shape[0]):
                    break
                cur_loc_mat = loc_ary[i_loc:i_loc+w,:]
                cur_loc_all = [cur_loc_mat[0,0], cur_loc_mat[-1,1]]

                #- obtain the label
                label = None
                if (true_mat is not None):
                    label = -1
                    multi_label = []
                    for i_ef in range(true_mat.shape[0]):
                        if (all(true_mat[i_ef,cur_loc_all[0]:cur_loc_all[1]])):
                            v = 1
                            label = i_ef
                        else:
                            v = -1
                        multi_label.append( v )
                    assert (np.sum(multi_label != -1) <= 1)

                #- compute the feature vector
                fvec_names, fvec = self.feature_extractor(pn_history, cur_loc_mat)
                fvec = [np.float32(x) for x in fvec]        # 32 bits to save the space

                out_row = [pn_num, cur_loc_all[0], cur_loc_all[1]] + fvec + [label]
                col_names = ['pn_num', 'loc_from', 'loc_to'] + fvec_names + ['label']
                out_table.append(out_row)
                pass
            pass
        return out_table, col_names
        
        

        

