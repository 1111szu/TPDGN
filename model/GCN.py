#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=66):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(
            in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.matmul(self.att, input)
        support = (1-alpha)*hi+alpha*h0
        r = support
        # ipdb.set_trace()
        output = theta*torch.matmul(support, self.weight)+(1-theta)*r
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=66):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(
            in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(
            in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)

        self.act_f = nn.Tanh()

    def forward(self, x, h0, lamda, alpha, l):

        y = self.gc1(x, h0, lamda, alpha, l)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, h0, lamda, alpha, l)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, input_feature, hidden_feature, p_dropout, nlayers=2, node_n=66, lamda=0.5, alpha=0.1):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks 
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.nlayers = nlayers
        self.fc1 = nn.Linear(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(nlayers):
            self.gcbs.append(
                GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        
        self.gcbs = nn.ModuleList(self.gcbs)
        self.fc7 = nn.Linear(hidden_feature, input_feature)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.lamda = lamda
        self.alpha = alpha
        self.cumu = nn.Parameter(torch.full((nlayers,), 0.8))

    def forward(self, x, history=None, is_out_resi=True):
        ret_history = []
        y = self.fc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)

        h0 = y.clone()

        y = self.do(y)

        for i in range(self.nlayers):
            if history is not None:
                y = self.cumu[i] * y + (1 - self.cumu[i]) * history[i]
            y = self.gcbs[i](y, h0, self.lamda, self.alpha, i+1)
            ret_history.append(y.clone())

        y = self.fc7(y)
        if is_out_resi:
            y = y + x
        return y, ret_history
