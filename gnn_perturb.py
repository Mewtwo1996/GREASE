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
        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
    def forward(self, x, adj):
        self.sub_adj = adj
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry
        #print("P is set")
        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True
        A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes)
        D_tilde = get_degree_matrix(A_tilde).detach()
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        if self.with_relu:
            x = F.relu(self.gc1(x, norm_adj))
        else:
            x = self.gc1(x, norm_adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, norm_adj)
        #x=self.gc3(x, adj)
        #return F.log_softmax(x, dim=1)
        return x

    def forward_prediction(self, x):
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions

        self.P = (torch.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat


        A_tilde = self.P * self.sub_adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)

        if self.with_relu:
            x = F.relu(self.gc1(x, norm_adj))
        else:
            x = self.gc1(x, norm_adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, norm_adj)
        return F.log_softmax(x, dim=1), self.P,self.P_hat_symm
    def loss(self,output,user_id, item_id):
        output = output.unsqueeze(0)
        cf_adj = self.P * self.adj
        cf_adj.requires_grad = True
        loss_graph_dist = sum(sum(abs(cf_adj - self.sub_adj))) / 2


