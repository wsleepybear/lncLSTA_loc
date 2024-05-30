# This is a pytorch implementation of biLSTM
from torch import nn
from torch.nn import functional as F
import utils
import pandas as pd
from model.transformer_ls import *
import pytorch_lightning as pl

from dataclasses import dataclass, field

import Config


class embedding(pl.LightningModule):
    def __init__(self, args):
        super(embedding, self).__init__()
        self.embed = nn.Embedding.from_pretrained(
            utils.loademb(args),
        )

    def forward(self, x):
        x = [self.embed(i.long()).unsqueeze(0) for i in x]
        return x


class CNN_Pool(pl.LightningModule):
    def __init__(self, filter_num, kernel_size,layer,spp_size,):
        super(CNN_Pool, self).__init__()
        self.spp_size=spp_size
        self.cnn_list=nn.ModuleList()
        self.pool_list=nn.ModuleList()
        for i in range(layer-1):
            self.cnn_list.append(nn.LazyConv1d(out_channels=filter_num*2, kernel_size=kernel_size))
            self.pool_list.append(nn.AdaptiveMaxPool1d(spp_size*(2**(layer-i-1))))
        self.cnn_list.append(nn.LazyConv1d(out_channels=filter_num, kernel_size=kernel_size))
        self.pool_list.append(nn.AdaptiveMaxPool1d(spp_size))
        self.batchnorm = nn.LazyBatchNorm1d()
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(0.5)
    def forward(self, x):
        # x: batchSize × seqLen × dim

        for j in range(len(self.cnn_list)):
            x_list=[]
            x = [self.cnn_list[j](i.transpose(2, 1)) for i in x]
            if j!=len(self.cnn_list)-1:
                for i in x :
                    if i.shape[2]>self.spp_size*(2**(len(self.cnn_list)-j-1)):
                        x_list.append(self.relu(self.pool_list[j](i).transpose(1, 2)))
                    else:
                        x_list.append(self.relu(i.transpose(1, 2)))
                x=x_list
            else :
                for i in x :
                    if i.shape[2]>self.spp_size*(2**(len(self.cnn_list)-j-1)):
                        x_list.append(self.pool_list[j](i).transpose(1, 2))
                    else:
                        x_list.append(i.transpose(1, 2))
                x=x_list

        x = torch.cat(x, dim=0)
        x = self.batchnorm(x)
        # x=self.relu(x)
        x=self.drop(x)
        return x


class textcnn(pl.LightningModule):
    def __init__(self, filter_num, contextSizeList, dropout):
        super(textcnn, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.LazyConv1d(out_channels=filter_num,
                                  kernel_size=contextSizeList[i]),
                    nn.GELU(),
                    nn.AdaptiveMaxPool1d(1)
                )
            )
        self.conv1dList = nn.ModuleList(moduleList)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # => scaleNum * (batchSize × filterNum)
        x = [conv(x).squeeze(dim=2) for conv in self.conv1dList]
        return self.dropout(torch.cat(x, dim=1))



class SPP(pl.LightningModule):
    def __init__(self, spp_size):
        super(SPP, self).__init__()
        self.spp = nn.AdaptiveMaxPool1d(spp_size)
        self.batchnorm = nn.LazyBatchNorm1d()

    def forward(self, x):
        x = [self.spp(i.transpose(2, 1)).transpose(1, 2)for i in x]
        x = torch.cat(x, dim=0)
        x = self.batchnorm(x)
        return x




class lsformer(pl.LightningModule):
    def __init__(self, dim, heads, num_layer, window_size, r, attdrop, ffdropout):
        super(lsformer, self).__init__()
        self.ls_former = LongShortTransformer(
            dim=dim,
            heads=heads,
            depth=num_layer,
            dim_head=dim,
            window_size=window_size,
            r=r,
            ff_dropout=ffdropout,
            attn_dropout=attdrop,

        )

    def forward(self, x):
        x = self.ls_former(x)
        return x




class lsATT(pl.LightningModule):
    def __init__(self, dim, heads, window_size, r, attdrop):
        super(lsATT, self).__init__()
        self.ls_att = LongShortAttention(
            dim=dim,
            heads=heads,
            dim_head=dim,
            window_size=window_size,
            r=r,
            dropout=attdrop
        )

    def forward(self, x):

        x = self.ls_att(x)
        # x=x1+x

        return x


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0.5, L=1):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = []
        list_FC_layers.append(nn.Linear(input_dim, 128*2, bias=True))
        list_FC_layers.append(nn.Linear(128*2, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.dropout = nn.Dropout(dropout)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.dropout(y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class BiLSTM(pl.LightningModule):
    def __init__(self, input, output, dropout):
        super(BiLSTM, self).__init__()
        self.BiLSTM = nn.LSTM(
            input_size=input,
            hidden_size=output//2,
            num_layers=6,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x, _ = self.BiLSTM(x)
        return x


class lncLSTA(pl.LightningModule):
    def __init__(self,):
        super(lncLSTA, self).__init__()

        self.args = Config.parse_args()
        self.embed = embedding(self.args)
        self.cnn = CNN_Pool(self.args.cnn1_num, self.args.cnn1_kernel,3,self.args.spp_size)
        self.cnn2 = CNN_Pool(self.args.cnn1_num, self.args.cnn2_kernel,3,self.args.spp_size)

        self.lsatt1 = lsATT(self.args.cnn1_num, self.args.lsatt1_head,
                             self.args.lsatt1_window_size, self.args.lsatt1_r, self.args.attdrop)
        self.lsatt2 = lsATT(self.args.cnn1_num, self.args.lsatt1_head,
                             self.args.lsatt1_window_size, self.args.lsatt1_r, self.args.attdrop)

        self.lstm = BiLSTM(self.args.cnn1_num,
                           self.args.cnn1_num, self.args.dropout)
        self.lstm2=BiLSTM(self.args.cnn1_num,
                           self.args.cnn1_num, self.args.dropout)
        self.lastcnn1 = textcnn(
            self.args.fc1in, self.args.contextSizeList, self.args.convdrop)
        self.lastcnn2 = textcnn(
            self.args.fc1in, self.args.contextSizeList, self.args.convdrop)
        self.lastcnn3 = textcnn(
            self.args.fc1in, self.args.contextSizeList, self.args.convdrop)
        self.lastcnn4 = textcnn(
            self.args.fc1in, self.args.contextSizeList, self.args.convdrop)
        # self.lastcnn3 = lastcnn3(
        #     self.args.fc3in, self.args.contextSizeList, self.args.convdrop)
        self.pool = nn.AdaptiveMaxPool1d(64)
        self.sigmod = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fc = nn.Sequential(
            nn.Linear((self.args.fc1in*6), self.args.fc1in ),
            nn.Dropout(0.15),
            nn.LazyLinear(5)
        )
        self.fc1=nn.LazyLinear(5)

        self.dropout = nn.Dropout(self.args.dropout)
        self.batchnorm = nn.LazyBatchNorm1d()
        self.layernorm = nn.LayerNorm(128)
    def forward(self, seqs):
        features = self.embed(seqs)

        features_1 = self.cnn(features)
        features_3 = self.cnn2(features)

        features1 = self.lstm(features_1)
        features2 = self.lsatt1(features_1)
        features3= self.lstm2(features_3)
        features4=self.lsatt2(features_3)

        features1 = self.lastcnn1(torch.cat((features1,features2),dim=2))
        features2 = self.lastcnn2(torch.cat((features3,features4),dim=2))

        features = torch.cat([features1, features2], dim=1)
        features = self.fc(features)
        # features=self.fc1(torch.cat([features1,output], dim=1))
        features = self.softmax(features)
        return features
    

