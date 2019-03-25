

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

import bi_lstm_model 
import cnn_model 
import entailment_model


class neighbor2vec (nn.Module): 
  def __init__(self,node_vector_dim_input,node_vector_dim_output): # @hidden_dim is dim of output after the CNN layer. 
    super(neighbor2vec, self).__init__()
    self.linear = nn.Linear(node_vector_dim_input,node_vector_dim_output) ## take vector as arg. Ax, so input has to be in batch mode
    xavier_uniform_ ( self.linear.weight )

  def forward( self, input) : # @input.unsqueeze(0) to make 3D
    ## take average of all the neighbors ( not max because each term has different num of go terms ?? )
    mean_val = torch.mean( self.linear (input.unsqueeze(0)), dim=1 )  # take average over each neighbor after transformation 
    return mean_val # @None to minimize many changes in code 
    # max_val, ind = torch.max( self.linear (input.unsqueeze(0)), dim=1 ) # take max over each neighbor after transformation 
    # return max_val, ind 

class neighbor2vec_parent_children (nn.Module): 
  def __init__(self,node_vector_dim_input,node_vector_dim_output): # @hidden_dim is dim of output after the CNN layer. 
    super(neighbor2vec_parent_children, self).__init__()
    self.parent2vec = neighbor2vec ( node_vector_dim_input,node_vector_dim_output ) ## conver all the parents into a vector 
    self.children2vec = neighbor2vec ( node_vector_dim_input,node_vector_dim_output )
  
  def forward(self,this_node_emb,parent_emb,children_emb): 
    ## @this_node_emb must be TRANSFORMED 
    parent = self.parent2vec (parent_emb)
    children = self.children2vec (children_emb)
    return F.relu( this_node_emb + parent + children )

class neighbor_1_layer (nn.Module) : 
  def __init__(self,node_vector_dim_input,node_vector_dim_output): # @hidden_dim is dim of output after the CNN layer. 
    super(neighbor_1_layer, self).__init__()
    self.linear_parent_transform = nn.Linear(node_vector_dim_input,node_vector_dim_output)
    self.linear_children_transform = nn.Linear(node_vector_dim_input,node_vector_dim_output)
    self.linear_sent_transform = nn.Linear( node_vector_dim_input, node_vector_dim_output )

  def forward(self, this_node_emb,parent_emb,children_emb):
    this_node_emb = self.linear_sent_transform ( this_node_emb )
    this_node_emb = this_node_emb + self.linear_parent_transform ( parent_emb )
    this_node_emb = this_node_emb + self.linear_children_transform ( children_emb )
    return this_node_emb # F.relu(
      

## for 2 layers neighbor model, need to get parents of parents 

def L2_neighbor_loss (label_emb,delta_matrix,scale=1) : 
  # delta_matrix = D - A 
  # label_emb is row=num_node, col=feature_dim 
  loss = label_emb.transpose(0,1) @ delta_matrix @ label_emb ## the @ is matrix mult. 
  return loss * scale 





