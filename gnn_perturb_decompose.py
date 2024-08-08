import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils1 import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix
import utils1
from utils1 import get_degree_matrix, normalize_adj, create_symm_matrix_from_vec, create_vec_from_symm_matrix

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dims, num_nodes, dropout=0.3, lr=0.001, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dims = dims
        self.num_nodes=num_nodes
        print("P")
        print("reset")
        self.reset_parameters()
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, dims, with_bias=with_bias)
        self.gc1.weight.requires_grad = False
        self.gc2.weight.requires_grad = False
        #self.gc3 = GraphConvolution(nhid, dims, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def reset_parameters(self, eps=10 ** -4):
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes
        self.P_sh = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.P_cf = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.P_fa = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.J=torch.ones(self.num_nodes,self.num_nodes)
        self.J.requires_grad = False
    def preprocess(self, target_user_id,target_item_id):
        self.J1=self.J
        self.J1[target_user_id,:]=0
        self.J1[target_user_id,:]=0
    def forward(self, x, adj):
        self.sub_adj = adj
        self.P_hat_symm_sh = create_symm_matrix_from_vec(self.P_sh, self.num_nodes)  # Ensure symmetry
        self.P_hat_symm_cf = self.J-(create_symm_matrix_from_vec(self.P_cf, self.num_nodes)+self. P_hat_symm_sh) 
        #self.P_hat_symm_fa = create_symm_matrix_from_vec(self.P_fa, self.num_nodes)  
        self.P_hat_symm_fa=self.J1+self. P_hat_symm_sh+create_symm_matrix_from_vec(self.P_fa, self.num_nodes)  
        #print("P is set")
        J=torch.ones(self.num_nodes)
        A_tilde_fa = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde_fa.requires_grad = True
        A_tilde_fa = F.sigmoid(self.P_hat_symm_fa) * self.sub_adj + torch.eye(self.num_nodes)
        A_tilde_cf = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde_cf.requires_grad = True
        A_tilde_cf = F.sigmoid(self.P_hat_symm_cf) * self.sub_adj + torch.eye(self.num_nodes)
        
        
        D_tilde_cf = get_degree_matrix(A_tilde_cf).detach()
        D_tilde_exp_cf = D_tilde_cf ** (-1 / 2)
        D_tilde_exp_cf[torch.isinf(D_tilde_exp_cf)] = 0
        norm_adj_cf = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        D_tilde_fa = get_degree_matrix(A_tilde_fa).detach()
        D_tilde_exp_fa = D_tilde_fa ** (-1 / 2)
        D_tilde_exp_fa[torch.isinf(D_tilde_exp_fa)] = 0
        norm_adj_fa = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        if self.with_relu:
            x_cf = F.relu(self.gc1(x, norm_adj_cf))
        else:
            x_cf = self.gc1(x_cf, norm_adj_cf)

        x_cf = F.dropout(x_cf, self.dropout, training=self.training)
        x_cf = self.gc2(x_cf, norm_adj)
        
         if self.with_relu:
            x_fa = F.relu(self.gc1(x, norm_adj_fa))
        else:
            x_fa = self.gc1(x_cf, norm_adj_fa)

        x_fa = F.dropout(x_fa, self.dropout, training=self.training)
        x_fa = self.gc2(x_fa, norm_adj)
        return x_cf,x_fa

    def forward_prediction(self, x):
        self.P_cf = (torch.sigmoid(self.P_hat_symm_cf) >= 0.5).float()  # threshold P_hat
        self.P_fa = (torch.sigmoid(self.P_hat_symm_fa) >= 0.5).float()  # threshold P_hat
        self.P_sh = (torch.sigmoid(self.P_hat_symm_sh) >= 0.5).float()  # threshold P_hat
        #print("P is set")
        J=torch.ones(self.num_nodes)
        A_tilde_fa = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde_fa.requires_grad = True
        A_tilde_fa = F.sigmoid(self.P_fa) * self.sub_adj + torch.eye(self.num_nodes)
        A_tilde_cf = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde_cf.requires_grad = True
        A_tilde_cf = F.sigmoid(self.P_cf) * self.sub_adj + torch.eye(self.num_nodes)
        
        
        D_tilde_cf = get_degree_matrix(A_tilde_cf).detach()
        D_tilde_exp_cf = D_tilde_cf ** (-1 / 2)
        D_tilde_exp_cf[torch.isinf(D_tilde_exp_cf)] = 0
        norm_adj_cf = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        D_tilde_fa = get_degree_matrix(A_tilde_fa).detach()
        D_tilde_exp_fa = D_tilde_fa ** (-1 / 2)
        D_tilde_exp_fa[torch.isinf(D_tilde_exp_fa)] = 0
        norm_adj_fa = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        if self.with_relu:
            x_cf = F.relu(self.gc1(x, norm_adj_cf))
        else:
            x_cf = self.gc1(x_cf, norm_adj_cf)

        x_cf = F.dropout(x_cf, self.dropout, training=self.training)
        x_cf = self.gc2(x_cf, norm_adj)
        
         if self.with_relu:
            x_fa = F.relu(self.gc1(x, norm_adj_fa))
        else:
            x_fa = self.gc1(x_cf, norm_adj_fa)

        x_fa = F.dropout(x_fa, self.dropout, training=self.training)
        x_fa = self.gc2(x_fa, norm_adj)
        return F.log_softmax(x_cf, dim=1), F.log_softmax(x_fa, dim=1) self.P_cf,self.P_fa,self.P_sh
    def loss(self,output,user_id, item_id):
        output = output.unsqueeze(0)
        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True
        loss_graph_dist = sum(sum(abs(cf_adj - self.sub_adj))) / 2


