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
from gnn_perturb import GCN
from scipy.sparse import csr_matrix
import scipy
import warnings
from register import dataset
from utils1 import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix
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

"""
index=9
target_user=testUser[index]
target_item=testItem[index]
"""


target_table=np.array(pd.read_csv("Testtarget.csv",header=None))
U=target_table[:,0]
I=target_table[:,1]+num_users
index_list=np.random.choice(len(U), 166, replace=False)
#index=553 #188
R_CFs=[]
R_acc=[]
Iter=0
index_record=[]
for index in index_list:
    index_record.append(index)
    print("Item",Iter)
    Iter+=1
    target_user=U[index]
    target_item=I[index]



    q=[]
    n_hop=2
    r_user=set()
    r_item=set()
    r=set()
    q.append((target_user,0))
    q.append((target_item,0))
    ori_rank=0
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

    ori_user=Recmodel.embedding_user.weight
    ori_item=Recmodel.embedding_item.weight
    output_user,output_item=LGCN.computer(ori_user,ori_item,Graph)
    sub_user_num=len(r_user)  #number of user in subgraph
    sub_item_num=len(r_item)  #numbr of item in subgraph
    ###extract subgraph
    r1=list(r_user)+list(r_item)
    Sub_tr = G_tr.subgraph(r1)  ##Undirected
    ##relabel
    Sub_tr=Sub_tr.to_undirected()
    edges0=list(Sub_tr.edges)
    mapping=dict(zip(r1,np.arange(len(r1))))
    mapping_inv=dict(zip(np.arange(len(r1)),r1))
    target_user_id=mapping[target_user]
    target_item_id=mapping[target_item]
    Sub_tr_re = nx.relabel_nodes(Sub_tr, mapping)
    edges1=list(Sub_tr_re.edges)
    S_adj=nx.adjacency_matrix(Sub_tr_re)
    S_adj=S_adj.astype("float")

    edges=np.array(Sub_tr.edges)

    #get the original features
    user_tr=list(r_user)
    item_tr=list(np.array(list(r_item))-num_users)  ##item_tr已经减掉了
    s_ori_user=ori_user[user_tr,:].cpu().detach().numpy()
    s_out_user=output_user[user_tr,:].cpu().detach().numpy()
    s_ori_item=ori_item[item_tr,:].cpu().detach().numpy()
    s_out_item=output_item[item_tr,:].cpu().detach().numpy()
    allPos = dataset.getUserPosItems([target_user])     ##allPos是所有train中的购买记录
    groundTrue = [dataset.testDict[target_user]]             ##Ground是所有test中的购买记录
    batch_users_gpu = torch.Tensor([target_user]).long()

    ##搜networkx from edge list to adj matrix
    features=np.vstack((s_ori_user,s_ori_item))
    label=np.vstack((s_out_user,s_out_item))
    device = 'cpu'
    S_adj1,features1,label1=utils1.preprocess(S_adj, features, label)
    #pre_out=S_model.forward()
    #train_label= np.random.choice(np.arange(features1.shape[0]), size=500, replace=False)
    #utils1.is_sparse_tensor(S_adj1)
    S_adj1 = S_adj1.to(device)
    features1 = features1.to(device)
    label1 = label1.to(device)

    from Gnn_analysis import fit_surrogate

    fit_surrogate(index)

    from gnn_perturb import GCN
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    from torch.nn.utils import clip_grad_norm

    num_node=S_adj1.shape[0]
    cf_model = GCN(nfeat=64, dims=64,num_nodes=num_node,
                        nhid=800, dropout=0.8, weight_decay=5e-4, device=device)
    cf_model.load_state_dict(torch.load("GCN.pkl"),strict=False)

    ###no grad
    for name, param in cf_model.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            param.requires_grad = False


    lr=0.1
    cf_optimizer = optim.SGD(cf_model.parameters(), lr=lr)
    loss_record=[]
    se=[]
    ls=[]
    record=[]
    action=[]
    #Recmodel.NoModify()
    cost=0
    flag=0
    item_index_ori=target_item-num_users
    Recmodel.NoModify()
    rating3 = Recmodel.getUsersRatingfromModified(batch_users_gpu)
    rating3[0, allPos[0]] = -(1 << 10)
    _, rating_K = torch.topk(rating3, k=20)
    rating_K_np=rating_K.cpu().detach().numpy()[0]
    ori_rank=np.argwhere(rating_K_np==item_index_ori)
    old_rank=ori_rank
    for e in range(360):
        if(flag==1):
            break
        #cf_optimizer.zero_grad()
        output = cf_model.forward(features1, S_adj1)
        output_actual,P,P_h= cf_model.forward_prediction(features1)
        loss1=torch.dot(output[target_user_id],output[target_item_id])
        cf_adj = P * S_adj1
        cf_adj.requires_grad = True
        loss_graph_dist = sum(sum(abs(cf_adj - S_adj1))) / 2
        #loss_total=-loss1-loss_graph_dist*0.3
        loss_total = loss1 + loss_graph_dist * 0.001
        ls.append(round(float(loss_total.cpu().detach().numpy()),2))
        loss_total.backward()
        clip_grad_norm(cf_model.parameters(), 10.0)
        cf_optimizer.step()
        diff=P.cpu().detach().numpy()*S_adj1.cpu().detach().numpy()-S_adj1.cpu().detach().numpy()
        itemindex=np.argwhere(diff==-1)
        if(len(itemindex)>0):
            se.append(itemindex)
            print("epoch",e)
            mod_id=[]
            #mod_id=[mapping_inv[itemindex[0]],mapping_inv[itemindex[1]]]
            #print("index",itemindex)
            for item in list(itemindex):
                if(list(item) not in record and item[0]<item[1] and item[0]<sub_user_num and item[1]>=sub_user_num):
                    cost+=1
                    record.append(list(item))
                    Recmodel.MGraph(mapping_inv[item[0]], mapping_inv[item[1]]-num_users)
                    rating3 = Recmodel.getUsersRatingfromModified(batch_users_gpu)
                    rating3[0, allPos[0]] = -(1 << 10)
                    _, rating_K = torch.topk(rating3, k=20)
                    print(rating_K)
                    rating_K_np = rating_K.cpu().numpy()[0]
                    new_rank = np.argwhere(rating_K_np == item_index_ori)
                    if (target_item-num_users) not in rating_K:
                        print("cost",cost)
                        print(record)
                        flag=1
                        break
                    print("new", new_rank, "old", old_rank)

                    if (new_rank < old_rank):
                        print("undo")
                        print("new",new_rank,"old",old_rank)
                        Recmodel.MGraph(mapping_inv[item[0]], mapping_inv[item[1]] - num_users)
                        cost -= 1
                        record.pop()
                    old_rank=max(old_rank,new_rank)
    if(flag==1):
        R_acc.append(1)
        R_CFs.append(record)
    else:
        R_acc.append(0)
        R_CFs.append([10000])
    print("ItemItemItem", Iter)
    print(np.sum(R_acc) / len(R_acc))
    R_acc_np = np.array(R_acc)
    R_CFs_np = np.array(R_CFs)
    index_record_np=np.array(index_record)
    np.save("acc.npy", R_acc_np)
    np.save("CFS.npy", R_CFs_np)
    np.save("Index.npy",index_record_np)


print(np.sum(R_acc)/len(R_acc))

R_acc=np.array(R_acc)
R_CFs=np.array(R_CFs)

np.save("acc.npy",R_acc)
np.save("CFS.npy",R_CFs)