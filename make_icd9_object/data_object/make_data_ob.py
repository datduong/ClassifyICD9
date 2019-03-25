from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle, gzip, os, sys, re
import random 
from random import shuffle
import numpy as np 
import pandas as pd

sys.path.append ( '/u/flashscratch/d/datduong/ICD9multitask/' )
import helper

import data_ob 


os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

def submitJobs ( k_val, batch_size, add_full_tree, do_manual, do_small ) : 
  ## load the vocab:index that has ICD words in it 
  vocab_index_map = pickle.load(open('vocab+icd_index_map.pickle','rb'))
  label_index_map = pickle.load(open("label_index_map.pickle","rb") )

  print ('test if padding is there')
  print (vocab_index_map['PADDING'])

  if k_val > 0: 
    """ should we filter to top k most freq ?? """
    print ('\n\ntesting on subset ??\n')
    
    if add_full_tree == 1: 
      # to_use = ['401.9','38.93','428.0','427.31','414.01']
      add_name = "_full_tree_"+str(k_val)+"icd"
      to_use = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_top_full_tree_"+str(k_val)+".pickle","rb"))
    else: 
      if do_manual == 1: 
        add_name = "_manual_"+str(k_val)+"icd"   
        to_use = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_manual_"+str(k_val)+".pickle","rb"))
        if do_small == 'do_small': 
          add_name = "_manual_small_"+str(k_val)+"icd" 
          to_use = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_manual_small_"+str(k_val)+".pickle","rb"))
      else: 
        add_name = "_"+str(k_val)+"icd"    
        to_use = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_top"+str(k_val)+".pickle","rb"))

    print ( to_use ) 

    label_index_map = pickle.load(open("label_index_map.pickle","rb"))
    select_label = [ label_index_map[k] for k in to_use if k in label_index_map ] ## may not have the correct icd?? 
    print ( select_label )
    print ('total valid labels {}'.format(len(select_label)))
    pickle.dump ( select_label, open("index_label_to_use_in_prediction"+add_name+".pickle","wb")) 


  ## !! keep @num_of_label as "large number", because it captures all the labels used. 
  num_of_label = 8996 # len(to_use) # 8996 # len(label_index_map) 
  print ('\ntotal num label saw being used in data {}\n\n'.format(num_of_label))
  if k_val == 0: 
    select_label = None # np.arange ( 0,num_of_label).tolist()
    add_name = "" ## not add anything to name 

  if add_full_tree == 1: 
    num_of_label = np.max (select_label) + 1 ## add 1 for correct indexing 
    print ('\nmust update if we want to add ancestors, new num. label {}\n'.format(num_of_label))

  if select_label is not None: 
    print ('\ntotal num label to be used in prediction {}\n\n'.format(len(select_label)))


  for data_type in ['test','dev','train']: # 

    main_data = '/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/cnn_att_data_Jan2019_batch/'+data_type+'_8996'+add_name+'.gzip' 

    if not os.path.exists( main_data ): 
      data_in = data_ob.Data ( '/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/disch_'+data_type+'_split_correct_icd'+add_name+'.csv', num_of_label, vocab_index_map, label_index_map, index_subset=select_label )
      pickle.dump ( data_in,  gzip.open('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/cnn_att_data_Jan2019_batch/'+data_type+'_8996'+add_name+'.gzip','wb') ) 
    else: 
      data_in = pickle.load ( gzip.open (main_data,"rb") ) 

    ## make batch 
    people = list ( data_in.note_for_obs.keys() ) 
    num_people = len(people)
    batch = {}
    batch_counter = 0 
    for k in range(0,num_people,batch_size): 
      # append 
      k2 = k+batch_size
      if k2 > num_people: ## make sure we get the last chunk correctly 
        k2 = num_people
      
      word_indexing = []
      if select_label is not None: 
        one_hot = torch.zeros( (k2-k, len(select_label) ) ) ## num people x label 
      else: 
        one_hot = torch.zeros( (k2-k, num_of_label ) )

      counter = 0 
      for p in people[k:k2]: 
        word_indexing.append ( data_in.note_for_obs[p] )
        one_hot[counter] = data_in.label_onehot_for_obs [p].unsqueeze(0)
        counter = counter + 1 

      # do padding 
      seq_lengths, seq_tensor, perm_idx = helper.pad_sentences(word_indexing, do_sort=True)
      one_hot = one_hot[ perm_idx ]
      # add to batch 
      batch[batch_counter] = {'input_sequence_indexing': seq_tensor,
                              'true_label': one_hot,
                              'sequence_indexing_len': seq_lengths}
      batch_counter = batch_counter + 1

    pickle.dump ( batch, gzip.open('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/cnn_att_data_Jan2019_batch/'+data_type+'_8996_b'+str(batch_size)+add_name+'.gzip','wb') ) 




if len(sys.argv)<1: ## run script 
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] ) 
	

