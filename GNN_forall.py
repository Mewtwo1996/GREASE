import world
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
import scipy
import utils1
import warnings
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
G_tr2=G_tr.to_undirected()
dict_of_tr = nx.to_dict_of_lists(G_tr)

A1 = nx.adjacency_matrix(G_tr)
A1=scipy.sparse.csr_matrix.toarray(A1)
A2 = nx.adjacency_matrix(G_tr2)
A2=scipy.sparse.csr_matrix.toarray(A2)
(A1==A1.T).all()
path="../data/lastfm"
file="lgn-lastfm-3-64.pth.tar"
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
load_weight_file = os.path.join(world.LOAD_PATH,file)
Recmodel.load_state_dict(torch.load(load_weight_file,map_location=torch.device('cpu')))

ori_user=Recmodel.embedding_user.weight
ori_item=Recmodel.embedding_item.weight
output_user,output_item=LGCN.computer(ori_user,ori_item,Graph)

S_adj=nx.adjacency_matrix(G_tr2)
S_adj=S_adj.astype("float")
ori_user=ori_user.detach().numpy()
output_user=output_user.detach().numpy()
ori_item=ori_item.detach().numpy()
output_item=output_item.detach().numpy()


features=np.vstack((ori_user,ori_item))
label=np.vstack((output_user,output_item))
device = 'cpu'
S_adj1,features1,label1=utils1.preprocess(S_adj, features, label)
#pre_out=S_model.forward()
train_label= np.random.choice(np.arange(features1.shape[0]), size=500, replace=False)
#utils1.is_sparse_tensor(S_adj1)
S_adj1 = S_adj1.to(device)
features1 = features1.to(device)
label1 = label1.to(device)
S_model = GCN(nfeat=64, dims=64,
                    nhid=64, dropout=0.5, weight_decay=5e-4, device=device)
S_model = S_model.to(device)
S_model.fit(features1, S_adj1, label1, train_label)


