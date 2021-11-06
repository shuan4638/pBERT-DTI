import numpy as np 
import torch
import sklearn
import glob
import pandas as pd
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import math

class PairWiseAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(PairWiseAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads // 2
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)
        scores = torch.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        return output

    def forward(self, x, mask=None):
        bs = x.size(0)
        x_ = self.layer_norm(x)
        k = self.k_linear(x_).view(bs, -1, self.h, 2, self.d_k)
        q = self.q_linear(x_).view(bs, -1, self.h, 2, self.d_k)
        v = self.v_linear(x_).view(bs, -1, self.h, 2, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        output = self.attention(q, k, v).view(bs, self.d_model) + x
        return output.squeeze(-1)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.linear_d =  nn.Sequential(nn.Linear(d_model//2, d_model), 
                        nn.ReLU(), 
                        nn.Dropout(0.2),
                        nn.Linear(d_model, d_model//2))
        self.linear_t =  nn.Sequential(nn.Linear(d_model//2, d_model), 
                        nn.ReLU(), 
                        nn.Dropout(0.2),
                        nn.Linear(d_model, d_model//2))
        self.layer_norm_d = nn.LayerNorm(d_model//2, eps=1e-6)
        self.layer_norm_t = nn.LayerNorm(d_model//2, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        drug_feats, target_feats = x.split(1024, -1)
        drug_feats, target_feats = self.layer_norm_d(drug_feats), self.layer_norm_t(target_feats)
        drug_feats, target_feats = self.linear_d(drug_feats), self.linear_t(target_feats)
        output = torch.cat([drug_feats, target_feats], -1) + x
        return output 

class PairwiseAttentionLinear(nn.Module):
    def __init__(self, d_model = 2048, heads = 8, n_layers = 3, dropout = 0.1):
        super(PairwiseAttentionLinear, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        if n_layers > 1:
            
            for _ in range(n_layers-1):
                att_stack.append(PairWiseAttention(heads, d_model, dropout))
                pff_stack.append(FeedForward(d_model, dropout=dropout))
        att_stack.append(PairWiseAttention(heads, d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        if n_layers > 1:
            self.pff_stack = nn.ModuleList(pff_stack)
        self.out =  nn.Sequential( nn.Linear(d_model, d_model//2), 
                        nn.ReLU(), 
                        nn.Dropout(0.2),
                        nn.Linear(d_model//2, 1))
        
    def forward(self, x):
        # x = self.embedding(x)
        if self.n_layers > 1:
            for n in range(self.n_layers-1):
                x = self.att_stack[n](x)
                x = self.pff_stack[n](x)
        x = self.att_stack[-1](x)
        
        return self.out(x)