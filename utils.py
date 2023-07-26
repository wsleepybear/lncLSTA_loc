# This module inlcudes functions that will be used in many other modules.
from ast import Return
from lib2to3.pgen2.literals import simple_escapes
import logging
# import word2vec
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sklearn
import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
import pickle
import os
import random
import subprocess
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class FocalCrossEntropyLoss(nn.Module):#focal loss
    def __init__(self, gama=2, weight=None, logit=False):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight, dtype=torch.float32) if weight is not None else weight
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        return -((1-P)**self.gama * torch.log(P)).mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha=3, gamma=8, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduce=False,)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        # F_loss =BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def init_logger():#初始化日志
    cur_file_name='train'
    cur_dir='/data/wk/'
    logger = logging.getLogger(cur_file_name)
    logger.setLevel(logging.INFO)

    # set two handlers
    fileHandler = logging.FileHandler(os.path.join(cur_dir, "{}.log".format(cur_file_name)))
    fileHandler.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    
    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    
    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger

#评价指标
class Metric(object):
    def __init__(self,output,label):
        self.output = output   #prediction label matric
        self.label = label    #true  label matric
    def average_precision(self,):
        y_pred =self.output
        y_true = self.label
        precision = average_precision_score(y_true, y_pred,average="micro")
        return precision
    def accuracy_subset(self,threash=0.5):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        accuracy=accuracy_score(y_true,y_pred)
        return accuracy
    
    def accuracy_mean(self,threash=0.5):
        y_pred =self.output
        y_true = self.label      
        y_pred=np.where(y_pred>threash,1,0)
        accuracy=np.mean(np.equal(y_true,y_pred))
        return accuracy
    
    def accuracy_multiclass(self):
        y_pred =self.output
        y_true = self.label     
        accuracy=accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
        return accuracy
    
    def micfscore(self,threash=0.5,type='micro'):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        return f1_score(y_pred,y_true,average=type)
    def macfscore(self,threash=0.5,type='macro'):
        y_pred =self.output
        y_true = self.label
        y_pred=np.where(y_pred>threash,1,0)
        return f1_score(y_pred,y_true,average=type)
    
    
    def fscore_class(self,type='macro'):
        y_pred =self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred,1),np.argmax(y_true,1),average=type)
    

    def miauc(self):
        y_pred =self.output
        y_true = self.label
        ROC = 0
        try:
            ROC = roc_auc_score(y_true, y_pred, average='micro', sample_weight=None)
        except:
            pass
        return ROC
    
    def maauc(self):
        y_pred =self.output
        y_true = self.label
        ROC = 0
        try:
            ROC = roc_auc_score(y_true, y_pred, average='macro', sample_weight=None)
        except:
            pass
        return ROC
    

def read_fea_file(fea_file):
    """read features"""
    fea_dict = {}

    f = open(fea_file, 'r')
    for line in f.readlines():
        line = line.strip()
        if len(line.split()) < 5:
            continue
        fea_dict[line.split()[0]] = [float(x) for x in line.split()[1:]]
    f.close()

    return fea_dict
def record(path,miAUC,maAUC,acc,f1):
    csv=pd.read_csv(path+'/record.csv')
    csv.loc[csv.shape[0]]=[miAUC,maAUC,acc,f1]
    csv.to_csv(path+'/record.csv',index=None)

def collate_fn(data_list):

    code = ([torch.tensor(i['seq']) for i in data_list],)
    label = ([i['label'] for i in data_list],
             [i['onehot'] for i in data_list])

    return code, label

def loademb(args):
    method=args.method
    if method=="char2vec":
        with open(f'cache/char2vec_k{args.k}_d{args.emb_dim}.pkl', 'rb') as f:
        # self.seq2emb = pickle.load(f)
            embedding = pickle.load(f)
            
    elif method=="one_hot":
        embedding=np.array([[0.25,0.25,0.25,0.25,0,0,0,0],
                            [1,0,0,0,0.1260,1,1,1],
                            [0,1,0,0,0.1335,0,0,1],
                            [0,0,1,0,0.1340,0,1,0],
                            [0,0,0,1,0.0806,1,0,0]])
    return torch.tensor(embedding).float()

def fold_rna_from_file( filepath,fold_algo='rnaplfold', probabilistic=True, **kwargs):
    assert (fold_algo in ['rnafold', 'rnasubopt', 'rnaplfold'])
    filepath=filepath
    data = pd.read_csv(filepath)
    all_seq=data['Sequence']

    # compatible with already computed structures with RNAfold
    prefix = '%s_%s_' % (fold_algo, probabilistic)
    if fold_algo == 'rnaplfold' or fold_algo == 'rnashapes':
        prefix += '%d_' % (kwargs.get('w', 200))
    if kwargs.get('modify_leaks', False):
        prefix = 'modified_' + prefix


    winsize = kwargs.get('w', 200)
    print('running rnaplfold with winsize %d' % (winsize))
    sp_rel_matrix = []
    sp_prob_matrix = []
    for i in tqdm(all_seq):
        res=fold_seq_rnaplfold(i,w=200,l=200,cutoff=1e-2, no_lonely_bps=True)
        sp_rel_matrix.append(res[0])
        sp_prob_matrix.append(res[1])
    pickle.dump(sp_rel_matrix,
                open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'wb'))
    pickle.dump(sp_prob_matrix,
                open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'wb'))

    print('Parsing', filepath, 'finished')
def seq_vec(seq):
    one2id,id2one = {"A":1,"T":2,"C":3,"G":4},["A","T","C","G"]
    seq_vec = torch.tensor([[one2id[i] for i in s] for s in seq])
    return seq_vec.squeeze().unsqueeze(0)
def fold_seq_rnaplfold(seq, w, l, cutoff, no_lonely_bps):
    np.random.seed(random.seed())
    name = str(np.random.rand())
    # Call RNAplfold on command line.
    no_lonely_bps_str = ""
    if no_lonely_bps:
        no_lonely_bps_str = "--noLP"
    try:
        cmd = 'echo %s | RNAplfold -W %d -L %d -c %.4f --id-prefix %s %s' % (seq, w, l, cutoff, name, no_lonely_bps_str)
        ret = subprocess.call(cmd, shell=True)
    except:
        pass
    # assemble adjacency matrix
    name += '_0001_dp.ps'
    start_flag = False
    if os.path.exists(name):
        row_col, link, prob = [], [], []
        length = len(seq)
        for i in range(length):
            if i != length - 1:
                row_col.append((i, i + 1))
                link.append(1)
                prob.append(1.)
            if i != 0:
                row_col.append((i, i - 1))
                link.append(2)
                prob.append(1.)
        # Extract base pair information.

        with open(name) as f:
            for line in f:
                if start_flag:
                    values = line.split()
                    if len(values) == 4:
                        source_id = int(values[0]) - 1
                        dest_id = int(values[1]) - 1
                        avg_prob = float(values[2])
                        # source_id < dest_id
                        row_col.append((source_id, dest_id))
                        link.append(3)
                        prob.append(avg_prob ** 2)
                        row_col.append((dest_id, source_id))
                        link.append(4)
                        prob.append(avg_prob ** 2)
                if 'start of base pair probability data' in line:
                    start_flag = True
        # delete RNAplfold output file.
        os.remove(name)
    else:
        # assemble adjacency matrix
        row_col, link, prob = [], [], []
        length = len(seq)
        for i in range(length):
            if i != length - 1:
                row_col.append((i, i + 1))
                link.append(1)
                prob.append(1.)
            if i != 0:
                row_col.append((i, i - 1))
                link.append(2)
                prob.append(1.)
        # delete RNAplfold output file.

    # placeholder for dot-bracket structure
    return (sp.csr_matrix((link, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),
            sp.csr_matrix((prob, (np.array(row_col)[:, 0], np.array(row_col)[:, 1])), shape=(length, length)),)

def draw_curve(para_dict, output_dir):
    sns.set_style("ticks")
    plt.figure(figsize=(8, 4), dpi=300,clear=True)
    df = pd.DataFrame(para_dict)
    sns.lineplot(data=df)
    plt.xlabel('Epoch')
    plt.savefig(output_dir)
    plt.close('all')
    
def tokenize_seq(seq, k=4, stride=1):
    """ Tokenizes the RNA sequence with k-mers.

    Args:
        seg: RNA sequence to be tokenized.
        k: length of the token.
        stride: step when moving the k window on sequence.

    Returns:
        tokens: tokenized sequence.
    """
    seq_length = len(seq)
    tokens = ""

    while seq_length > k:
        tokens += seq[-seq_length:-seq_length+k] + " "
        seq_length -= stride
    tokens += seq[-k:]

    return tokens



def split_seq(seq, k, stride):
    """ Split nucleotide sequence.

    The original sequence is nucleotides whose bases are A, T, C and G. This
    function split it with k-pts and stride stride.

    Example:
        >>> split_seq('ATCGATCG', k=4, stride=2)
        ['ATCG', 'CGAT', 'ATCG']

    Args:
        seq: nucleotides sequence, which is supposed to be a string.
        k: length of nucleotide combination. E.g. k of 'ATCGAT' should be 6.
        stride: step when moving the k window on sequence.

    Returns:
        splited_seq: list containing splited seqs.

    """
    seq_length = len(seq)
    splited_seq = []

    while seq_length > k:
        splited_seq.append(seq[-seq_length:-seq_length+k])
        seq_length -= stride
    splited_seq.append(seq[-k:])

    return splited_seq




