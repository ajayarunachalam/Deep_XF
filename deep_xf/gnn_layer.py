#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 12/10/2021
Goal: Graph Neural Network Layers
Version: 0.0.1
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def device_as(x,y):
	return x.to(y.device)

# tensor operations now support batched inputs
def calc_degree_matrix_norm(a):
	return torch.diag_embed(torch.pow(a.sum(dim=-1),-0.5))

def create_graph_lapl_norm(a):
	size = a.shape[-1]
	a +=  device_as(torch.eye(size),a)
	D_norm = calc_degree_matrix_norm(a)
	L_norm = torch.bmm( torch.bmm(D_norm, a) , D_norm )
	return L_norm

class GCN_Layer(nn.Module):
		"""
		A simple GCN layer, similar to https://arxiv.org/abs/1609.02907
		"""
		def __init__(self, in_features, out_features, bias=True):

			super().__init__()
			self.linear = nn.Linear(in_features, out_features, bias=bias)

		def forward(self, X, A):
				"""
				A: adjecency matrix
				X: graph signal
				"""
				L = create_graph_lapl_norm(A)
				x = self.linear(X)
				return torch.bmm(L, x)