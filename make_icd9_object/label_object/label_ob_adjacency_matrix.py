from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import sys, os, pickle

import numpy as np
import pickle
import gzip
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


## 
sys.path.append("/u/flashscratch/d/datduong/ICD9multitask/")
import helper 

try: 
  os.mkdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')
except: 
  pass 

os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')


# description_in_indexing_arr = pickle.load( open("description_in_indexing_arr.pickle","rb") )
# seq_lengths, seq_tensor, perm_idx = helper.pad_sentences(description_in_indexing_arr, do_sort=False)
# label_data = {} 
# label_data['label_indexing'] = seq_tensor
# label_data['label_name'] = pickle.load(open("label_name_arr.pickle","rb"))
# label_data['label_len'] = torch.FloatTensor ( seq_lengths.numpy() ) 
# pickle.dump( label_data , open('label_data.pickle','wb'))

# exit() 

# parents/children look up index 
label_index_map = pickle.load( open("label_index_map.pickle","rb") )
reverse_label_index_map = pickle.load( open("reverse_label_index_map.pickle","rb") )

parents = pickle.load ( open("parent_icd.pickle","rb") ) 
children = pickle.load ( open("children_icd.pickle","rb") ) 


nei = {'parent':{}, 'children':{}}
for i in reverse_label_index_map.keys(): ## @i is numeric 1 2 3 ..  
  this_icd = reverse_label_index_map[i]
  if this_icd in parents: ## has parents 
    nei['parent'][this_icd] = [label_index_map[k] for k in parents[this_icd]] ## get the index of the parent terms 
  if this_icd in children: ## has children 
    nei['children'][this_icd] = [label_index_map[k] for k in children[this_icd]]

#
pickle.dump (nei, open("neighbor_lookup.pickle","wb") ) 

## example: 

# print ('this_icd {}'.format(this_icd))
# print (nei['parent'][this_icd])
# print (nei['children'][this_icd])


## create adjadcency matrix ... ?? too big ?? 

counter = 0 
num_label = len(label_index_map)
adjacency = np.zeros( (num_label,num_label) ) ## do this by col , so col-1 is 1-hot of labels which are its parents 
for this_icd in label_index_map: ## only care about what has parents 
  if this_icd in nei['parent']: 
    adjacency [ label_index_map[this_icd], nei['parent'][this_icd] ] = 1.0 / len(nei['parent'][this_icd])
    if counter == 0: ## see example 
      print (this_icd)
      print (adjacency [ label_index_map[this_icd], nei['parent'][this_icd] ])
      print (adjacency [ label_index_map[this_icd]].sum())
      print (nei['parent'][this_icd])
      counter = 1 


pickle.dump( adjacency, gzip.open("adjacency_parent.gzip.pickle","wb") )
print ('see sum {}'.format(adjacency.sum()))

## children nodes 

counter = 0 
adjacency = np.zeros( (num_label,num_label) ) ## do this by col , so col-1 is 1-hot of labels which are its parents 
for this_icd in label_index_map: ## only care about what has parents 
  if this_icd in nei['children']: 
    adjacency [ label_index_map[this_icd] , nei['children'][this_icd] ] = 1.0 / len(nei['children'][this_icd])
    if counter == 0: ## see example 
      print (this_icd)
      print (adjacency [ label_index_map[this_icd], nei['children'][this_icd] ])
      print (adjacency [ label_index_map[this_icd]].sum())
      print (nei['children'][this_icd])
      counter = 1

pickle.dump( adjacency, gzip.open("adjacency_children.gzip.pickle","wb") )
print ('see sum {}'.format(adjacency.sum()))


# ## create diff matrix A
adjacency_parent = pickle.load( gzip.open("adjacency_parent.gzip.pickle","rb") )
adjacency_children = pickle.load( gzip.open("adjacency_children.gzip.pickle","rb") )

## !! @adjacency is the children_adjacency so we don't need to load 
# notice @np.ceil so that we count full connection, not weighted 
adjacency_children = np.ceil(adjacency_parent) + np.ceil(adjacency_children)  ## should have no overlap 
degree = np.sum ( adjacency_children, axis=1 ) ## sum by row 
print (degree)
delta_matrix = np.diag( degree ) - adjacency_children
print ('see sum {}'.format(delta_matrix.sum()))

pickle.dump( delta_matrix, gzip.open("delta_matrix.gzip.pickle","wb") )


# /u/home/d/datduong/project/anaconda3/bin/python3 /u/flashscratch/d/datduong/ICD9multitask/make_icd9_object/label_object/label_ob.py 
