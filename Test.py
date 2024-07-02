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
from torch import nn
from scipy.sparse import csr_matrix
import warnings
from register import dataset
import copy

#Recmodel = register.MODELS[world.model_name](world.config, dataset)
device='cpu'
dataset = dataloader.LastFM()
num_users  = dataset.n_users
num_items  = dataset.m_items
Graph =dataset.getSparseGraph()
path="../data/lastfm"
trainData = pd.read_table(join(path, 'data1.txt'), header=None)
testData  = pd.read_table(join(path, 'test1.txt'), header=None)
trainData-= 1
testData -= 1
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

index=8
target_user=testUser[index]
target_item=testItem[index]

###check whether correct
d_user=dataset.trainUser
d_item=dataset.trainItem
allPos = dataset.getUserPosItems([target_user])
np.array(dict_of_tr[target_user])-num_users
d_item[d_user==target_user]




path="../data/lastfm"
file="lgn-lastfm-3-64.pth.tar"
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
load_weight_file = os.path.join(world.LOAD_PATH,file)
Recmodel.load_state_dict(torch.load(load_weight_file,map_location=torch.device('cpu')))

ori_user=Recmodel.embedding_user.weight
ori_item=Recmodel.embedding_item.weight
output_user,output_item=LGCN.computer(ori_user,ori_item,Graph)

allPos = dataset.getUserPosItems([target_user])
groundTrue = [dataset.testDict[target_user]]
batch_users_gpu = torch.Tensor([target_user]).long()
rating1 = Recmodel.getUsersRating(batch_users_gpu)   ##from light gcn

target_users_emb = output_user[batch_users_gpu.long()]   #(batch,emb)
items_emb = output_item  #(all_item,emb) #(all_item,emb)
f = nn.Sigmoid()
rating2 = f(torch.matmul(target_users_emb, items_emb.t()))  ##by hand

##correct

exclude_index = []
exclude_items = []
for items in allPos:
    exclude_items.extend(items)

rating2[0, exclude_items] = -(1<<10)
_, rating_K = torch.topk(rating2, k=20)###item对应于rating2的下标
rating = rating2.cpu().numpy()

print("rating_K",rating_K)
print("Ground",groundTrue )
print(set(np.squeeze(rating_K.numpy())) & set(np.squeeze(groundTrue)))

G_LGCN=dataset.Graph


