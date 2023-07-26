from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import utils
import Config
import os
import pickle
from gensim.models import Word2Vec
import scipy.sparse as sp
from utils import *
class RNAdataset(Dataset):
    def __init__(self,dir) -> None:
        super(RNAdataset,self).__init__()
        self.args = Config.parse_args()
        data = pd.read_csv(dir)

        self.seq=data['Sequence']
        self.label=data['SubCellular_Localization']
        self.onehot,self.num_label=self.class2num(self.label) 
        self.vectorize(method=self.args.method,emb_dim=self.args.emb_dim)

    def __len__(self):
        return len(self.label) 
    
    def __getitem__(self, index):
        item={}
        item['seq']=self.seq_vec[index]
        # item['tab_fea']=self.feature_list[index]
        item['onehot']=torch.tensor(self.onehot[index])
        item['label']=torch.tensor(self.num_label[index])
        return item

    def vectorize(self,method="char2vec", emb_dim=64, window=5, sg=1, 
                        workers=8):
        if method=='char2vec':
            self.all_seq=[]
            for i in self.seq:
                self.all_seq.append(utils.split_seq(
                i, k=self.args.k, stride=self.args.stride))
            self.kmers2id,self.id2kmers = {"<EOS>":0},["<EOS>"]
            self.kmer=self.args.k
            self.emb_dim=emb_dim
            kmersCnt = 1
            for rna in tqdm(self.all_seq):
                for kmers in rna:
                    if kmers not in self.kmers2id:
                        self.kmers2id[kmers] = kmersCnt
                        self.id2kmers.append(kmers)
                        kmersCnt += 1
            self.kmersNum = kmersCnt
            if os.path.exists(f'cache/{method}_k{self.kmer}_d{emb_dim}.pkl'):
                with open(f'cache/{method}_k{self.kmer}_d{emb_dim}.pkl', 'rb') as f:
                    # self.seq2emb = pickle.load(f)
                    self.char2vec = pickle.load(f)
            else:
                seq = [i+['<EOS>'] for i in self.all_seq]
                model = Word2Vec(seq, min_count=0, window=window, vector_size=emb_dim, workers=workers, sg=sg, epochs=10)
                self.char2vec = np.zeros((self.kmersNum, emb_dim), dtype=np.float32)
                # self.seq2emb = {}
                for i in range(1,self.kmersNum):
                    self.char2vec[i] = model.wv[self.id2kmers[i]]
                    # self.seq2emb[self.id2kmers[i]] = model.wv[self.id2kmers[i]]
                print("seq vector dimension: {}".format(emb_dim))
                with open(f'cache/{method}_k{self.kmer}_d{emb_dim}.pkl', 'wb') as f:
                    pickle.dump(self.char2vec, f, protocol=4)
            self.seq_vec= np.array([[self.kmers2id[i] for i in s] for s in self.all_seq])     
        elif method=='one_hot':
            self.one2id,self.id2one = {"A":1,"T":2,"C":3,"G":4},["A","T","C","G"]
            self.seq_vec = np.array([[self.one2id[i] for i in s] for s in self.seq])
    def class2num(self,label):
        label_name=[]
        label_id=[]
        for i in label:
            if len(label_name)<5 and i not in label_name:    
                label_name.append(i)
                label_id.append(len(label_id))
        label2num=dict(zip(label_name,label_id))
        num=label.map(label2num)
        label2onehot = dict(zip(label_name,np.eye(len(label_id))))
        onehot=label.map(label2onehot)
        return onehot,num
    def loadgraph(self,filepath,fold_algo='rnaplfold',probabilistic=True,w=200,):
        prefix = '%s_%s_' % (fold_algo, probabilistic)
        prefix += '200_'
        if os.path.exists(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix))):
            pass
        else:
            fold_rna_from_file(filepath)
        sp_rel_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}rel_mat.obj'.format(prefix)), 'rb'))
        sp_prob_matrix = pickle.load(open(os.path.join(os.path.dirname(filepath), '{}prob_mat.obj'.format(prefix)), 'rb'))
        adjacency_matrix = np.array(sp_rel_matrix)
        probability_matrix = np.array(sp_prob_matrix)
        matrix = (adjacency_matrix, probability_matrix)

        adj_list=[]
        prob_list=[]
        for i in range(len(matrix[0])):
            link=matrix[0][i]
            prob=matrix[1][i]
            adj = sp.coo_matrix(link)
            indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
            adj = torch.LongTensor(indices)
            adj_list.append(adj)
            prob_list.append(prob.data) 
        self.adj_list=adj_list
        self.prob_list=prob_list