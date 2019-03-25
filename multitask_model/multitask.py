


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle, os, time, datetime, gzip 
import numpy as np
import pandas as pd

from copy import deepcopy

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
import neighbor_model 

sys.path.append("/local/datdb/ICD9multitask/cnn_att")
import cnn_att_nei

sys.path.append("/local/datdb/ICD9multitask/spen_model")
import spen_model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# main_path = "/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019"
main_path = "/local/datdb/MIMIC3database/format10Jan2019/"
os.chdir (main_path)

permutation_choice = pickle.load (open("permutation_choice.pickle","rb"),encoding = 'latin1')

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
  print ('total label to train/test on {}'.format(len(select_label)))
  print (select_label)
else: 
  add_name = "" 
  select_label = np.arange(0,8996)


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
label_layer_dim = args.label_layer_dim

do_gcnn = args.do_gcnn 
init_label_emb = args.init_label_emb

if args.model_cnn is None: 
  cnn_mod = cnn_att_nei.cnn_att_nei_model(num_of_labels,num_of_word,word_vec_dim,num_of_filter,width_of_filter,label_encoded_dim,gcnn_dim,batch_size,do_gcnn,init_label_emb)
  cnn_mod.load_pretrain_emb(pretrain_emb)
else: 
  print ('\nload {}'.format(args.model_cnn))
  cnn_mod = torch.load( args.model_cnn )
 
if args.not_train_w2v_emb: 
  cnn_mod.embedding.weight.requires_grad = False
  print ('\nmust replace the word emb, these will be carried over from last cnn_mod save ??')
  print ('train word emb ? true/false, also make sure they update them correctly in the spen_model')
  print (cnn_mod.embedding.weight.requires_grad)

## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION
cnn_mod.eval() 

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


cnn_mod.new_label_emb = None ## if load gcnn, we don't care about this if @do_gcnn is true.  
if args.do_gcnn: 
  print ('load gcnn, we will now create the new label emb')
  ## @self.label_emb will be created inside this class
  cnn_mod.do_forward_label(label_data, adjacency_parent, adjacency_children, new_label_emb=None) ## create the labels to be used once for all forward step. ... save time. 
  # new_label_emb = cnn_mod.new_label_emb
  adjacency_parent = None ## reset ?? 
  adjacency_children = None 
  # new_label_emb = cnn_mod.new_label_emb
  print ('see gcnn emb')
  print (cnn_mod.new_label_emb)

for param in cnn_mod.parameters():
  param.requires_grad = False  


print('now init the spen model')
torch.cuda.empty_cache()

if args.model_load is None: 
  spen_mod = spen_model.StructureEnergyLoss(cnn_mod,num_of_labels,label_layer_dim,args.do_label_label,args.L2_scale, epoch_y=args.epoch_y, epoch_predict_y=args.epoch_predict_y, lr_optim_y=args.lr_optim_y, lr_predict_y=args.lr_predict_y, wait_until_y=80, epoch_params=args.epoch_params, lr_params=args.lr, batch_size=batch_size, do_gcnn=args.do_gcnn, init_label_emb=args.init_label_emb, load_gcnn_emb=args.load_gcnn_emb,not_train_w2v_emb=args.not_train_w2v_emb, not_update_cnn_att=args.not_update_cnn_att,weight_term=args.weight_term,lr_lagran=args.lr_lagran,pair_wise_only=args.pair_wise_only)
else: 
  spen_mod = torch.load(args.model_load)
  ## we have to update the step size ?? 
  spen_mod.lr_predict_y = args.lr_predict_y
  spen_mod.epoch_predict_y = args.epoch_predict_y
  spen_mod.lr_optim_y = args.lr_optim_y
  spen_mod.epoch_y = args.epoch_y
  spen_mod.lr_params = args.lr
  spen_mod.epoch_params = args.epoch_params
  spen_mod.wait_until_y = 80
  spen_mod.lr_lagran = args.lr_lagran
  


print ('model is')
print (spen_mod)

spen_mod.cuda() 

if not os.path.exists(args.result_folder): 
  os.mkdir (args.result_folder)

## store last best guess ?? 
## !! CALL THE CNN MODEL, BECAUSE IT'S POSSIBLE FOR US TO UPDATE THE CNN DURING TRAINING SPEN. 
cnn_mod.eval() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION
last_best_guess_label = {}
cnn_best_label = {} 
cnn_prediction = None 
for i in range (len(valid_data)): 
  seq_tensor = valid_data[i]['input_sequence_indexing'].cuda()
  seq_lengths = valid_data[i]['sequence_indexing_len']
  cnn_best = F.sigmoid ( cnn_mod.do_forward(label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb) ) 
  cnn_best_label [i] = deepcopy ( cnn_best.cpu().data.numpy() ) 
  cnn_prediction = helper.append_predicted_value ( cnn_best, cnn_prediction )
  if args.warm_start: 
    last_best_guess_label [i] = deepcopy ( cnn_best.cpu().data.numpy() )
  else: 
    last_best_guess_label[i] = None 


#
last_best_guess_label_on_train = {}
cnn_best_label_on_train = {} 
for i in range (len(train_data)): 
  seq_tensor = train_data[i]['input_sequence_indexing'].cuda()
  seq_lengths = train_data[i]['sequence_indexing_len']
  cnn_best = F.sigmoid ( cnn_mod.do_forward(label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb) ) 
  cnn_best_label_on_train [i] = deepcopy ( cnn_best.cpu().data.numpy() )
  if args.warm_start: 
    last_best_guess_label_on_train [i] = deepcopy ( cnn_best.cpu().data.numpy() )
  else: 
    last_best_guess_label_on_train[i] = None 


last_f1 = -np.inf 
last_auc = -np.inf 

if args.do_test: 
  train_range = []
  args.epoch = 1
else: 
  train_range = range ( len(train_data) )
  

for iter in range (args.epoch): 

  start = time.time()   
  # spen_mod.turn_on_training() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION

  train_range = np.random.permutation(train_range)
  for i in train_range: ## each batch len(ent_tr_data) len(train_data)

    spen_mod.turn_on_training() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION

    spen_mod.zero_grad() ## must zero out historical values 

    seq_tensor = train_data[i]['input_sequence_indexing'].cuda()
    seq_lengths = train_data[i]['sequence_indexing_len']
    true_label = train_data[i]['true_label'].cuda()
    
    if i in np.arange(10,15,1) : do_print = True
    else: do_print = False 

    # should we use constraint during training? or just use SGD?
    energy = spen_mod.train_1_batch ( label_data, seq_tensor,seq_lengths,true_label, select_label, adjacency_parent, adjacency_children, new_label_emb=spen_mod.feature_nn_model.new_label_emb, do_print=do_print, constraint=args.do_constraint, keep_feasible=args.keep_feasible )

    if i % 100 == 0: 
      torch.cuda.empty_cache()

    if do_print: 

      spen_mod.turn_on_eval() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION
      print ('\ntrain batch i={}'.format(i))

      if args.do_constraint: 
        best_guess_label = spen_mod.optim_new_sample_constraint (label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb, last_best_guess_label=last_best_guess_label_on_train[i],do_print=do_print, cnn_best_label=cnn_best_label_on_train[i], keep_feasible=args.keep_feasible)  
      else: 
        best_guess_label = spen_mod.optim_new_sample (label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb, last_best_guess_label=last_best_guess_label_on_train[i],do_print=do_print)  
      
      if args.start_last_best: ## if we use warm-start then, we will always start at the cnn-best 
        last_best_guess_label_on_train[i] = best_guess_label.cpu().data.numpy() ## update best ?? 

      xy_relation = spen_mod.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb )

      print ('\ncnn label energy')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, torch.FloatTensor(cnn_best_label_on_train[i]).cuda() , do_print=do_print )
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      print ('\n\nguess label energy fit back on train data')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, best_guess_label,do_print=do_print )
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      if args.brute_force:
        print ('\n\nbrute force label energy fit back on train data')
        best_brute_label = spen_mod.optim_new_sample_brute_force (label_data, seq_tensor, seq_lengths, select_label, new_label_emb=None, do_print=do_print, true_label=true_label, permutation_choice=permutation_choice ) 
        E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, best_brute_label, do_print=do_print )
        print (E_XYhat.transpose(0,1))
        print (E_XYhat.sum())

      print ('\n\ntrue label energy fit back on train data')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, true_label,do_print=do_print )
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      print ('\n\ncompare labels on train (cnn vs prediction vs brute vs true)')
      print (np.round ( cnn_best_label_on_train[i], 4 ))
      print (np.round(best_guess_label.cpu().data.numpy(),4))
      if args.brute_force:
        print (np.round(best_brute_label.cpu().data.numpy(),4))
      print (np.round(true_label.cpu().data.numpy(),1))


  print ('end iter')
  # print (spen_mod.energy_function.Label_Label_Energy.label_compat[0].weight)

  print('\niter '+ str(iter) +' time ' + str(time.time() - start) ) 

  spen_mod.turn_on_eval() ## MUST SAY TRAIN/EVAL TO ENABLE DROP.OUT IN TRAINING AND AVOID DROP.OUT IN EVALUATION

  if args.brute_force:
    brute_force_prediction = None 

  sgd_prediction = None
  input_label_energy = None 

  for i in range( len(valid_data) ): ## each batch len(ent_tr_data)

    seq_tensor = valid_data[i]['input_sequence_indexing'].cuda()
    seq_lengths = valid_data[i]['sequence_indexing_len']
   
    if i in np.arange(10,15,1) : do_print = True
    else: do_print = False 

    if args.do_constraint: 
      ## @last_best_guess_label[i] is init as cnn_best. @last_best_guess_label[i] is used as warm-start. 
      ## do we need to retain the last best labels ?? 
      best_guess_label = spen_mod.optim_new_sample_constraint (label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb, last_best_guess_label=last_best_guess_label[i],do_print=do_print,near_cnn=args.near_cnn,cnn_best_label=cnn_best_label[i], keep_feasible=args.keep_feasible)  
    else: 
      best_guess_label = spen_mod.optim_new_sample (label_data, seq_tensor,seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb, last_best_guess_label=last_best_guess_label[i],do_print=do_print,near_cnn=args.near_cnn,cnn_best_label=cnn_best_label[i]) 
   
    xy_relation = spen_mod.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb )
    if args.brute_force:
      best_brute_label = spen_mod.optim_new_sample_brute_force (label_data, seq_tensor, seq_lengths, select_label, new_label_emb=spen_mod.feature_nn_model.new_label_emb, do_print=False, true_label=None, permutation_choice=permutation_choice ) ## torch.FloatTensor(cnn_best_label[i]).cuda() valid_data[i]['true_label'].cuda()
      brute_force_prediction = helper.append_predicted_value ( best_brute_label, brute_force_prediction )

    sgd_prediction = helper.append_predicted_value ( best_guess_label, sgd_prediction )
    input_label_energy = helper.append_predicted_value ( xy_relation, input_label_energy )
    

    if args.start_last_best: ## if we use warm-start then, we will always start at the cnn-best 
      ## must do this, otherwise we may not converge correctly (seem to be very hard to converge in later iterations)
      last_best_guess_label[i] = deepcopy ( best_guess_label.cpu().data.numpy() )

    if do_print: # do_print: 

      print ('dev batch {}'.format(i))
  
      print ('\ncnn label energy')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, torch.FloatTensor(cnn_best_label[i]).cuda() , do_print=do_print )
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      print ('\n\nguess label energy')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, best_guess_label, do_print=do_print )
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      if args.brute_force:
        print ('\n\nbrute force label energy (vs dev true label). should true be always lowest?')
        E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, best_brute_label, do_print=do_print )
        print (E_XYhat.transpose(0,1))
        print (E_XYhat.sum())

      print ('\n\ntrue label energy')
      E_XYhat = spen_mod.energy_function.forward_energy_only( xy_relation, valid_data[i]['true_label'].cuda(), do_print=do_print)
      print (E_XYhat.transpose(0,1))
      print (E_XYhat.sum())

      print ('\ncompare labels (cnn-best vs prediction vs brute vs true)')
      print (np.round ( cnn_best_label[i], 4 ))
      print (np.round ( best_guess_label.cpu().data.numpy(),4) )
      if args.brute_force:
        print (np.round ( best_brute_label.cpu().data.numpy(),4))
      print (np.round ( valid_data[i]['true_label'],1 ) ) 

 
  ## get evaluation metric
  torch.cuda.empty_cache()

  print ('\ncnn prediction')
  print (cnn_prediction)
  output_metric = evaluation_metric.all_metrics ( np.round(cnn_prediction), valid_truth, yhat_raw=cnn_prediction, k=args.top_k, input_label_energy=input_label_energy ) ##  [ 0:(16*3) , :]
  evaluation_metric.print_metrics( output_metric )

  print ('\nsgd-type prediction')
  print (sgd_prediction)
  output_metric = evaluation_metric.all_metrics ( np.round(sgd_prediction), valid_truth, yhat_raw=sgd_prediction, k=args.top_k, input_label_energy=input_label_energy ) ##  [ 0:(16*3) , :]
  evaluation_metric.print_metrics( output_metric )

  if not args.do_test and ((output_metric['auc_macro'] >= last_auc) or (output_metric['f1_macro'] >= last_f1)): 
    print ('save')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    print (timestamp)
    torch.save ( spen_mod, args.result_folder+"/current_best.pytorch" )
    last_f1 = output_metric['f1_macro']
    last_auc = output_metric['auc_macro']
    pickle.dump(sgd_prediction, open(args.result_folder+"/sgd_prediction.pickle","wb") )

  if args.brute_force:
    print ('\nbrute force prediction')
    print (brute_force_prediction)
    output_metric = evaluation_metric.all_metrics ( np.round(brute_force_prediction), valid_truth, yhat_raw=brute_force_prediction, k=args.top_k, input_label_energy=input_label_energy ) ##  [ 0:(16*3) , :]
    evaluation_metric.print_metrics( output_metric )

  print ('\ntrue label')
  print (valid_truth)

  # else: 
  #   ## not decrease.. do we reduce the LR ?? 
  #   if iter > 50:
  #     print ('update LR size')
  #     spen_mod.lr_params = spen_mod.lr_params * .9
  #     spen_mod.lr_optim_y = spen_mod.lr_optim_y * .9
  #     spen_mod.lr_predict_y = spen_mod.lr_predict_y * .9

