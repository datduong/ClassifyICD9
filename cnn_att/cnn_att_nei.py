

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

sys.path.append("../common_model")
import bi_lstm_model 
import cnn_model 
import neighbor_model 

class cnn_att_nei_model (nn.Module) : 
  def __init__(self,num_of_labels,num_of_word,word_vec_dim,num_of_filter,width_of_filter,label_encoded_dim,gcnn_dim,batch_size,do_gcnn,init_label_emb, do_gcnn_only=False): 
    
    super(cnn_att_nei_model, self).__init__()
    
    self.do_gcnn = do_gcnn 
    self.do_gcnn_only = do_gcnn_only
    self.init_label_emb = init_label_emb

    self.num_of_labels = num_of_labels
    if do_gcnn: 
      self.gcnn_dim = gcnn_dim ## dim of the label after being embeded. 
    else:
      self.gcnn_dim = 0 

    self.label_encoded_dim = word_vec_dim ## new dim of the labels, if we take ave. of words in the description, then label has same dim as @word_vec
    if self.init_label_emb: 
      self.label_encoded_dim = label_encoded_dim

    self.batch_size = batch_size

    # init these models 
    self.embedding = nn.Embedding(num_of_word, word_vec_dim) # word embedding 

    # num_of_filter = self.gcnn_dim + word_vec_dim # word_vec_dim ## because we take average of words 
    self.num_of_filter = num_of_filter # word_vec_dim
    self.prediction_model = cnn_model.cnn_att_model (num_of_labels,num_of_word,word_vec_dim,num_of_filter,width_of_filter,self.label_encoded_dim + self.gcnn_dim, self.do_gcnn, self.init_label_emb ) 
    
    ## @word_vec_dim because we take the average of word vec
    if do_gcnn: 
      self.neighbors_layer1 = neighbor_model.neighbor_1_layer(word_vec_dim,self.gcnn_dim) ## take vector as arg. Ax, so input has to be in batch mode
      self.neighbors_layer2 = neighbor_model.neighbor_1_layer(self.gcnn_dim,self.gcnn_dim)
   
  def load_pretrain_emb (self,pretrained_weight): # @pretrained_weight is pretrained emp in np format word x dim 
    self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

  def label_embedding (self, label_indexing, seq_lengths ): 
    ## labels are short enough, so we just take average of words
    desc_embedding = self.embedding (label_indexing) ## squeeze to make 2D ?? 
    return torch.sum(desc_embedding, 1).squeeze(1) / seq_lengths ## take average of words in the description, this broadcast the len division 

  def do_gcnn_layer (self, label_emb, adjacency_parent, adjacency_children) :

    ## taking mean is implicit if we scale @adjacency_matrix 
    parent = torch.matmul ( adjacency_parent , label_emb ) # .transpose(0,1)  
    children = torch.matmul ( adjacency_children , label_emb ) #.transpose(0,1) 
    label_emb = self.neighbors_layer1(label_emb , parent , children) ## pass through layer 1 

    ## pass through layer 2
    parent = torch.matmul ( adjacency_parent, label_emb ) #.transpose(0,1)  
    children = torch.matmul ( adjacency_children, label_emb ) #.transpose(0,1) 
    label_emb = self.neighbors_layer2(label_emb , parent , children) 

    return label_emb


  def loss_sum(self, label_data, adjacency_parent, adjacency_children, prediction_data, select_label, delta_matrix=None, new_label_emb=None):

    input_sequence_emb = self.embedding(prediction_data['input_sequence_indexing'].cuda()) ## embed for DNA or protein amino acid
  
    if (not self.init_label_emb) or self.do_gcnn or self.do_gcnn_only : 
      label_emb = self.label_embedding ( label_data['label_indexing'].cuda() , label_data['label_len'].unsqueeze(0).transpose(0,1).cuda() ) ## wording into embedding
    else: 
      label_emb = None

    if self.do_gcnn_only: 
      new_label_emb = self.do_gcnn_layer ( label_emb, adjacency_parent, adjacency_children )
      loss = new_label_emb.transpose(0,1) @ delta_matrix @ new_label_emb ## the @ is matrix mult.
      return torch.trace ( loss )  

    if self.do_gcnn: 
      if new_label_emb is None: 
        new_label_emb = self.do_gcnn_layer ( label_emb, adjacency_parent, adjacency_children )
      #
      loss = self.prediction_model.loss_function ( input_sequence_emb, prediction_data['sequence_indexing_len'], label_emb[select_label], prediction_data['true_label'].cuda(), new_label_emb[select_label] )

    else: 
      if not self.init_label_emb: 
        loss = self.prediction_model.loss_function ( input_sequence_emb, prediction_data['sequence_indexing_len'], label_emb[select_label], prediction_data['true_label'].cuda(), None )
      else: 
        loss = self.prediction_model.loss_function ( input_sequence_emb, prediction_data['sequence_indexing_len'], None, prediction_data['true_label'].cuda(), None )

    return loss

  def do_forward_label(self, label_data, adjacency_parent, adjacency_children, new_label_emb=None) : 
    
    if (not self.init_label_emb) or self.do_gcnn or self.do_gcnn_only :
      self.label_emb =  self.label_embedding ( label_data['label_indexing'].cuda() , label_data['label_len'].unsqueeze(0).transpose(0,1).cuda() )
    else: 
      self.label_emb = None

    if self.do_gcnn or self.do_gcnn_only:
      if new_label_emb is not None: 
        self.new_label_emb = new_label_emb
      else: 
        self.new_label_emb = self.do_gcnn_layer ( self.label_emb, adjacency_parent, adjacency_children )
    else:
      self.new_label_emb = None


  def do_forward(self, label_data, seq_tensor,seq_lengths, select_label, new_label_emb=None): ## @new_label_emb is meant for pre-trained
    ## do forward pass 

    ##!! must call @do_forward_label outside this function, it will save time. 
    input_sequence_emb = self.embedding(seq_tensor) ## embed for DNA or protein amino acid 
 
    if self.do_gcnn: 
      if new_label_emb is not None: 
        return self.prediction_model.forward ( input_sequence_emb, seq_lengths, self.label_emb[select_label], new_label_emb[select_label] )
      else: 
        return self.prediction_model.forward ( input_sequence_emb, seq_lengths, self.label_emb[select_label], self.new_label_emb[select_label] ) 
    else: 
      if not self.init_label_emb: 
        return self.prediction_model.forward ( input_sequence_emb, seq_lengths, self.label_emb[select_label], None ) 
      else :
        return self.prediction_model.forward ( input_sequence_emb, seq_lengths, None, None )
