


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle, os, time, datetime, gzip 
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_


sys.path.append("/local/datdb/ICD9multitask/")
import arg_input
args = arg_input.get_args()

print (args)

import helper
import evaluation_metric

sys.path.append("/local/datdb/ICD9multitask/common_model")
import bi_lstm_model 
import cnn_model 
import cnn_att_nei


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# main_path = "/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019"
main_path = "/local/datdb/MIMIC3database/format10Jan2019/"
os.chdir (main_path)

if args.do_gcnn or (not args.init_label_emb) : ## call the label layer 
  label_data = pickle.load(open("label_data.pickle","rb"))
else: 
  label_data = None 


# extract_index = pickle.load(open('neighbor_lookup.pickle',"rb"))
pretrain_emb = pickle.load(open("pubmed_go_data_vocab_indexing_prune_pretrain_emb_"+args.w2v_emb+".pickle","rb")) ## later used to load into the emb

add_name = args.add_name 

if add_name is not None: 
  print ('\n\ntesting on subset ??\n')
  # add_name = "_100icd"
  select_label = pickle.load ( open("index_label_to_use_in_prediction"+add_name+".pickle","rb")) 
  print (select_label)
else: 
  add_name = "" 
  select_label = np.arange(0,8996)

print ('load icd count partition')
index_icd_count_partition = pickle.load(open("index_icd_count_partition.pickle","rb"))


batch_size = args.batch_size

train_data = pickle.load(gzip.open("cnn_att_data_Jan2019_batch/train_8996_b"+str(batch_size)+add_name+".gzip","rb"))
valid_data = pickle.load(gzip.open("cnn_att_data_Jan2019_batch/dev_8996_b"+str(batch_size)+add_name+".gzip","rb"))
valid_truth = helper.make_ground_truth_np( valid_data , do_vstack=True)

print ('num batch to train {}'.format(len(train_data)))


num_of_labels = len(select_label) # 5 ## len( label_data['label_len'] )  # 9013 # @num_of_labels are the labels to be tested (possible will not have all the labels in database)
print ('num_of_labels {}'.format(num_of_labels))

num_of_word = pretrain_emb.shape[0]
word_vec_dim = pretrain_emb.shape[1]
num_of_filter = args.num_of_filter
width_of_filter = 10 
label_encoded_dim = args.label_encoded_dim
gcnn_dim = args.gcnn_dim 

do_gcnn = args.do_gcnn 
init_label_emb = args.init_label_emb
do_gcnn_only = args.do_gcnn_only

if args.model_load is None :
  cnn_mod = cnn_att_nei.cnn_att_nei_model(num_of_labels,num_of_word,word_vec_dim,num_of_filter,width_of_filter,label_encoded_dim,gcnn_dim,batch_size,do_gcnn,init_label_emb,do_gcnn_only)
  cnn_mod.load_pretrain_emb(pretrain_emb)
else: 
  cnn_mod = torch.load ( args.model_load )

if args.not_train_w2v_emb: 
  cnn_mod.embedding.weight.requires_grad = False

print ('train word emb ? true/false')
print (cnn_mod.embedding.weight.requires_grad)

print ('model is')
print (cnn_mod)

cnn_mod.cuda() 

optimizer = optim.Adam( filter(lambda p: p.requires_grad, cnn_mod.parameters()) , lr = args.lr )

# for p in cnn_mod.parameters(): 
#   if p.requires_grad == True: 
#     print (p.shape)

    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)

if args.do_gcnn: 
  adjacency_parent = pickle.load (gzip.open("adjacency_parent.gzip.pickle","rb"))
  print ('see sum connection parent {}'.format( np.sum(adjacency_parent) ))
  adjacency_children = pickle.load (gzip.open("adjacency_children.gzip.pickle","rb"))
  print ('see sum connection children {}'.format( np.sum(adjacency_children)) )
  adjacency_parent = torch.FloatTensor(adjacency_parent).cuda() ## do not transpose, the row is a node, columns are its connection to other nodes 
  adjacency_children = torch.FloatTensor(adjacency_children).cuda() 
  
else: 
  adjacency_parent = None
  adjacency_children = None 

if args.do_gcnn_only: 
  delta_matrix = pickle.load (gzip.open("delta_matrix.gzip.pickle","rb"))
  delta_matrix = torch.FloatTensor(delta_matrix).cuda() 
else:
  delta_matrix = None 


if not os.path.exists(args.result_folder): 
  os.mkdir (args.result_folder)

last_f1 = -np.inf 
do_gcnn_loss = np.inf 

for iter in range (args.epoch): 
  
  scheduler.step(iter)

  start = time.time()   
  cnn_mod.train() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION

  sum_loss = Variable(torch.zeros(1)).cuda()  

  if args.do_gcnn_only:  ## don't have anything to evaluate ?? 
    cnn_mod.zero_grad() ## must zero out historical values 
    model_loss = cnn_mod.loss_sum (label_data,adjacency_parent,adjacency_children,train_data[0],select_label,delta_matrix=delta_matrix) 
    model_loss.backward() 
    optimizer.step()
    sum_loss = model_loss / len(label_data)

    print ('\nloss for iter' + str(iter))
    print (sum_loss)
    end = time.time()
    print('\ntime ' + str(end - start) ) 
    if sum_loss < do_gcnn_loss: 
      print ('save')
      do_gcnn_loss = sum_loss 
      torch.save ( cnn_mod, args.result_folder+"/current_best.pytorch" )
      label_emb = cnn_mod.label_embedding ( label_data['label_indexing'].cuda() , label_data['label_len'].unsqueeze(0).transpose(0,1).cuda() ) ## wording into embedding
      new_label_emb = cnn_mod.do_gcnn_layer ( label_emb, adjacency_parent, adjacency_children )
      torch.save ( new_label_emb , args.result_folder+"/gcnn_icd_label_emb.pytorch" )
      print ('GCNN new label emb')
      print (new_label_emb)
    continue

  ## !! 

  for i in range( len(train_data) ): ## each batch len(ent_tr_data)

    cnn_mod.zero_grad() ## must zero out historical values 
    model_loss = cnn_mod.loss_sum (label_data,adjacency_parent,adjacency_children,train_data[i],select_label,delta_matrix=delta_matrix) 
    model_loss.backward() 
    optimizer.step()
    sum_loss = sum_loss + model_loss

  torch.cuda.empty_cache()
  print ('\nloss for iter' + str(iter))
  print (sum_loss)
  end = time.time()
  print('\ntime ' + str(end - start) ) 


  cnn_mod.eval() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION
  
  if args.do_gcnn_only or args.do_gcnn or (not args.init_label_emb) : ## call the label layer 
    cnn_mod.do_forward_label(label_data, adjacency_parent, adjacency_children) ## create the labels to be used once for all forward step. ... save time. 

  prediction_value = None
  for i in range( len(valid_data) ): ## each batch len(ent_tr_data)

    seq_tensor = valid_data[i]['input_sequence_indexing'].cuda()
    seq_lengths = valid_data[i]['sequence_indexing_len']
    true_label = valid_data[i]['true_label'].cuda()

    value = cnn_mod.do_forward(label_data, seq_tensor,seq_lengths, select_label) ## num_people x num_label for prediction 
    prediction_value = helper.append_predicted_value ( value, prediction_value )
  
  
  # torch.cuda.empty_cache()
  ## get evaluation metric
  prediction_value = sigmoid(prediction_value)
  print ('prediction')
  print (prediction_value)
  animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_value), valid_truth, yhat_raw=prediction_value, k=args.top_k ) ##  [ 0:(16*3) , :]
  evaluation_metric.print_metrics( animo_go_metric )

  if add_name == "":  
    print ('\nevaluate based on count')
    name = index_icd_count_partition.keys()
    name.sort() ## make it easier to compare 
    for count_type in name: 
      print ("\n"+count_type)
      where = index_icd_count_partition[count_type]
      animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_value[:,where]), valid_truth[:,where], yhat_raw=prediction_value[:,where], k=args.top_k )
      evaluation_metric.print_metrics( animo_go_metric )
      
  
  if animo_go_metric['auc_macro'] > last_f1: 
    print ('save')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    print (timestamp)
    torch.save ( cnn_mod, args.result_folder+"/current_best.pytorch" )
    last_f1 = animo_go_metric['auc_macro']



# if not args.do_test: 
#   exit() 

print ('\n\ndo testing')

valid_data = pickle.load(gzip.open("cnn_att_data_Jan2019_batch/test_8996_b"+str(batch_size)+add_name+".gzip","rb"))
valid_truth = helper.make_ground_truth_np( valid_data , do_vstack=True)
print ('num batch to test {}'.format(len(valid_data)))

print ('\nload back best model\n')
cnn_mod = torch.load ( args.result_folder+"/current_best.pytorch" )
cnn_mod.eval() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION

torch.cuda.empty_cache()

if args.do_gcnn_only or args.do_gcnn or (not args.init_label_emb) : ## call the label layer 
  cnn_mod.do_forward_label(label_data, adjacency_parent, adjacency_children) ## create the labels to be used once for all forward step. ... save time. 

prediction_value = None
for i in range( len(valid_data) ): ## each batch len(ent_tr_data)
  seq_tensor = valid_data[i]['input_sequence_indexing'].cuda()
  seq_lengths = valid_data[i]['sequence_indexing_len']
  true_label = valid_data[i]['true_label'].cuda()
  value = cnn_mod.do_forward(label_data, seq_tensor,seq_lengths, select_label) ## num_people x num_label for prediction 
  prediction_value = helper.append_predicted_value ( value, prediction_value )

## get evaluation metric
prediction_value = sigmoid(prediction_value)
print ('\nprediction')
print (prediction_value)
animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_value), valid_truth, yhat_raw=prediction_value, k=args.top_k ) ##  [ 0:(16*3) , :]
evaluation_metric.print_metrics( animo_go_metric )


pickle.dump(prediction_value,open(args.result_folder+"/prediction_on_test.pickle",'wb'))
pickle.dump(valid_truth,open(args.result_folder+"/true_on_test.pickle",'wb'))





# if add_name == "":  
#   print ('\nevaluate based on count')
#   name = list ( index_icd_count_partition.keys() ) 
#   name.sort() ## make it easier to compare 
#   for count_type in name: 
#     print ("\n"+count_type)
#     where = index_icd_count_partition[count_type]
#     animo_go_metric = evaluation_metric.all_metrics ( np.round(prediction_value[:,where]), valid_truth[:,where], yhat_raw=prediction_value[:,where], k=args.top_k )
#     evaluation_metric.print_metrics( animo_go_metric )
    
