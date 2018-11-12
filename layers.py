import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.

    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    def forward(self, matrix1, matrix2):
        self.save_for_backward(matrix1, matrix2)
        return torch.mm(matrix1, matrix2)

    def backward(self, grad_output):
        matrix1, matrix2 = self.saved_tensors
        grad_matrix1 = grad_matrix2 = None

        if self.needs_input_grad[0]:
            grad_matrix1 = torch.mm(grad_output, matrix2.t())

        if self.needs_input_grad[1]:
            grad_matrix2 = torch.mm(matrix1.t(), grad_output)

        return grad_matrix1, grad_matrix2

class InnerProductGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=False):
        super(InnerProductGraphConvolution, self).__init__()

        #pdb.set_trace()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.transferi = Parameter(torch.Tensor(in_features, in_features))
        self.transferj = Parameter(torch.Tensor(in_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        init_range = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        if self.bias is not None:
            self.bias.data.uniform_(-init_range, init_range)

        self.transferi.data.normal_(0.0, 0.001)
        self.transferj.data.normal_(0.0, 0.001)
        #torch.nn.init.eye_(self.transferi)
        #torch.nn.init.eye_(self.transferj)

    def forward(self, x):

        support = SparseMM()(x, self.weight)
        x_transferi = torch.mm(x, self.transferi) #1006*300
        #x_transferj = torch.mm(x, self.transferj) #1006*300
        x_transferj = torch.mm(x, self.transferi) #1006*300
        adj = torch.mm(x_transferi, x_transferj.t())
        #pdb.set_trace()
        adj = self.softmax(adj)

        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repre__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'

class MyGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=False):
        super(MyGraphConvolution, self).__init__()

        #pdb.set_trace()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.adj = Parameter(adj)
        self.adj.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        if self.bias is not None:
            self.bias.data.uniform_(-init_range, init_range)

    def forward(self, x):
        support = SparseMM()(x, self.weight)
        output = SparseMM()(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repre__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-init_range, init_range)
        if self.bias is not None:
            self.bias.data.uniform_(-init_range, init_range)

    def forward(self, x, adj):
        support = SparseMM()(x, self.weight)
        output = SparseMM()(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repre__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'


class LayerNormalization(Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = Parameter(torch.zeros(d_hid), requires_grad=True)
        #self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.a_2)
        torch.nn.init.kaiming_normal_(self.b_2)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        #pdb.set_trace()

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out
