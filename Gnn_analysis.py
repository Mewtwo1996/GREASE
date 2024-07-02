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
import copy
import warnings
from register import dataset
#Recmodel = register.MODELS[world.model_name](world.config, dataset)
def fit_surrogate(index):
    import numpy as np
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
    G_tr2=G_tr.to_undirected() ##G_tr 中仅包括train集中含有的user和item,light GCN中的是全部的，映射要谨慎
    dict_of_tr = nx.to_dict_of_lists(G_tr)

    A1 = nx.adjacency_matrix(G_tr)
    A1=scipy.sparse.csr_matrix.toarray(A1)
    A2 = nx.adjacency_matrix(G_tr2)
    A2=scipy.sparse.csr_matrix.toarray(A2)
    (A1==A1.T).all()

    #index=9
    #target_user=testUser[index]
    #target_item=testItem[index]
    target_table=np.array(pd.read_csv("Testtarget.csv",header=None))
    U=target_table[:,0]
    I=target_table[:,1]+num_users
    index=518 #188
    target_user=U[index]
    target_item=I[index]




    q=[]
    n_hop=2
    r_user=set()
    r_item=set()
    r=set()
    q.append((target_user,0))
    q.append((target_item,0))

    while(q):
        node,d=q.pop(0)
        r.add(node)
        if(node>=1892):
            r_item.add(node)
        else:
            r_user.add(node)
        if(node in dict_of_tr and d<n_hop):
            for n in dict_of_tr[node]:
                q.append((n,d+1))
    print("all",len(r),"user",len(r_user),"item",len(r_item))
    #print(len(record))


    path="../data/lastfm"
    file="lgn-lastfm-3-64.pth.tar"
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    load_weight_file = os.path.join(world.LOAD_PATH,file)
    Recmodel.load_state_dict(torch.load(load_weight_file,map_location=torch.device('cpu')))

    ori_user = Recmodel.embedding_user.weight
    ori_item = Recmodel.embedding_item.weight
    output_user, output_item = LGCN.computer(ori_user, ori_item, Graph)
    sub_user_num = len(r_user)  # number of user in subgraph
    sub_item_num = len(r_item)  # numbr of item in subgraph
    ###extract subgraph
    r1 = list(r_user) + list(r_item)
    Sub_tr = G_tr.subgraph(r1)  ##Undirected
    ##relabel
    Sub_tr = Sub_tr.to_undirected()

    Sub_node_index = list(np.array(Sub_tr.nodes))
    mapping_t = dict(zip(Sub_node_index, np.arange(len(r1))))
    index_shuffle = []
    for i in r1:
        index_shuffle.append(mapping_t[i])
    S_adj_t = nx.adjacency_matrix(Sub_tr)
    S_adj_t = S_adj_t.astype("float")
    S1 = S_adj_t[:, index_shuffle]
    S_adj = copy.deepcopy(S1[index_shuffle, :])

    mapping = dict(zip(r1, np.arange(len(r1))))
    mapping_inv = dict(zip(np.arange(len(r1)), r1))
    target_user_id = mapping[target_user]
    target_item_id = mapping[target_item]

    #get the original features
    user_tr=list(r_user)
    item_tr=list(np.array(list(r_item))-num_users)
    s_ori_user=ori_user[user_tr,:].cpu().detach().numpy()
    s_out_user=output_user[user_tr,:].cpu().detach().numpy()
    s_ori_item=ori_item[item_tr,:].cpu().detach().numpy()
    s_out_item=output_item[item_tr,:].cpu().detach().numpy()


    ##搜networkx from edge list to adj matrix
    features=np.vstack((s_ori_user,s_ori_item))
    label=np.vstack((s_out_user,s_out_item))
    device = 'cpu'
    S_adj1,features1,label1=utils1.preprocess(S_adj, features, label)
    #pre_out=S_model.forward()
    train_label= np.random.choice(np.arange(features1.shape[0]), size=int(len(r)*0.99), replace=False)
    #utils1.is_sparse_tensor(S_adj1)
    S_adj1 = S_adj1.to(device)
    features1 = features1.to(device)
    label1 = label1.to(device)
    S_model = GCN(nfeat=64, dims=64,
                        nhid=800, dropout=0.8, weight_decay=5e-4, device=device)
    S_model = S_model.to(device)
    r=S_model.fit(features1, S_adj1, label1, train_label)
    #S_adj_norm = utils1.normalize_adj_tensor(S_adj1)
    torch.save(S_model.state_dict(), "GCN.pkl")


    model1=GCN(nfeat=64, dims=64,
                        nhid=800, dropout=0.8, weight_decay=5e-4, device=device)

    model1.load_state_dict(torch.load("GCN.pkl"))

    """
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(r[:500]))
    y = np.array(r[:500])
    plt.xlabel("iter")
    plt.ylabel("error")
    plt.plot(x,y)
    plt.show()
    """