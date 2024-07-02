import world
import torch.nn.functional as F
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import networkx as nx
import utils1
import dataloader
from register import dataset
from dataloader import BasicDataset as BD
import world
import utils1
import scipy as sp
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
# ==============================
utils1.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
import os
import LGCN
from GCN_CLASS import GCN
from scipy.sparse import csr_matrix
from register import dataset
#Recmodel = register.MODELS[world.model_name](world.config, dataset)
device='cpu'
dataset = dataloader.LastFM()
num_users  = dataset.n_users
num_items  = dataset.m_items
Graph =dataset.getSparseGraph()
path="../data/lastfm"
trainData = pd.read_table(join(path, 'data1.txt'), header=None)
testData  = pd.read_table(join(path, 'test1.txt'), header=None)
trainUser = np.array(trainData[:][0])
trainUniqueUsers = np.unique(trainUser)  ##1878
trainItem = np.array(trainData[:][1])+num_users       # self.trainDataSize = len(self.trainUser)
trainUniqueItems = np.unique(trainItem)  ##4476
testUser  = np.array(testData[:][0])
testUniqueUsers = np.unique(testUser)  ##1858
testItem  = np.array(testData[:][1])+num_users
testUniqueItems = np.unique(testItem)

test_adj=np.stack([testUser, testItem])
test_adj_df = pd.DataFrame({'node_1':testUser, 'node_2':testItem})
train_adj=np.stack([trainUser,trainItem])
train_adj_df = pd.DataFrame({'node_1':trainUser, 'node_2':trainItem})
G_tr = nx.from_pandas_edgelist(train_adj_df, 'node_1', 'node_2', create_using=nx.Graph())
dict_of_tr = nx.to_dict_of_lists(G_tr)


edg0=list(G_tr.edges)
r1=[]
for i in range(len(edg0)):
    n0=edg0[i][0]
    n1=edg0[i][1]
    if((n1,n0) in edg0):
        continue
    else:
        r1.append((n0,n1))

A1 = nx.adjacency_matrix(G_tr)
A1=scipy.sparse.csr_matrix.toarray(A1)
(A1==A1.T).all()

def count(idx):
    return np.sum(idx)