from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, random, sys
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_

import numpy as np

sys.path.append('../')
import helper

sys.path.append('../common_model')
import cnn_model

# torch.manual_seed(1337)

class SquareActivation (nn.Module): 
  def __init__(self):
    super ( SquareActivation, self).__init__() 
  def forward(self,input): 
    return input**2


class ReduceLabelDim (nn.Module): ## this is almost like "label-label" energy, because we map f(label) = some_number
  def __init__(self,num_of_labels,label_layer_dim): # reduce original dim of labels into smaller space @new_num_of_labels
    super ( ReduceLabelDim, self).__init__()
    self.num_of_labels = num_of_labels
    self.label_layer_dim = label_layer_dim

    drop_out = 0.1
    if self.num_of_labels < self.label_layer_dim: 
      drop_out = .2

    self.standard_linear = nn.Sequential (
      nn.Linear( self.num_of_labels, self.label_layer_dim, bias=True ), ## next layer
      nn.Tanh(),
      nn.Dropout(p=drop_out), ## avoids overfit
      nn.Linear( self.label_layer_dim, self.label_layer_dim, bias=True ),
      nn.Tanh(),
      nn.Dropout(p=drop_out), ## avoids overfit
      nn.Linear( self.label_layer_dim, self.num_of_labels, bias=True ),
      nn.Tanh()
      ) ## so we can apply nn.Linear to each entry in the batch

    self.label_compat = nn.Sequential(
      nn.Linear( self.num_of_labels, self.label_layer_dim, bias=True ),
      nn.Softplus(),
      nn.Dropout(p=drop_out), ## avoids overfit
      nn.Linear(self.label_layer_dim, 1), # int(np.ceil(self.num_of_labels/2))
      nn.Softplus())

    xavier_uniform_ (self.standard_linear[0].weight)
    # xavier_uniform_ (self.standard_linear[2].weight)
    # xavier_uniform_ (self.standard_linear[4].weight)
    
    xavier_uniform_ (self.label_compat[0].weight)
    # xavier_uniform_ (self.label_compat[2].weight)

  def forward(self,label) :
    new_label = self.standard_linear(label)
    new_label = self.label_compat ( new_label )
    return new_label


class LabelDeltaL2 (nn.Module): ## L2 loss
  def __init__(self, scale=1):
    super(LabelDeltaL2, self).__init__()
    self.scale = scale ## acts like a weight to scale if L2 loss is important

  def forward(self,true_label,guess_label):
    var = true_label - guess_label ## row-wise vectors
    var = torch.sum ( var*var, dim=1 ) * self.scale  ## batch mode, num_batch x 1 x 1
    return var.unsqueeze(1) ## make into batch_size x 1 


class PairWiseLoss (nn.Module): ## in the form y^t A y (this is like conditional random field)
  def __init__(self,num_of_labels): # reduce original dim of labels into smaller space @new_num_of_labels
    super ( PairWiseLoss, self).__init__()
    self.num_of_labels = num_of_labels
    self.PairMatrixA = nn.Linear(num_of_labels,num_of_labels,bias=False)
  
  def label_compat( self, label ) :
    ## label is in batch mode
    ## @label is batch_size x num_label, must conver to 3D for broadcast multiplication 
    label = label.unsqueeze(1).transpose(1,2) # 3D
    compat = self.PairMatrixA.weight.matmul(label) # matrix mul. 
    # transpose back label.transpose(1,2), so we have row vector x col vec 
    compat = label.transpose(1,2) @ compat ## multiply, can it does batch ?? 
    compat = compat.squeeze(1) ## want output as num_batch x 1 (col vector)
    return compat
 

class Energy_Function (nn.Module):
  def __init__(self,num_of_labels,label_layer_dim,do_label_label=True,scale=1,weight_term=1,pair_wise_only=False): # input,label

    super(Energy_Function, self).__init__()

    self.scale = scale
    self.weight_term = weight_term
    self.label_layer_dim = label_layer_dim
    self.num_of_labels = num_of_labels 
    self.do_label_label = do_label_label
   
    self.pair_wise_only = pair_wise_only

    if not pair_wise_only: 
      self.Label_Label_Energy = ReduceLabelDim (self.num_of_labels, self.label_layer_dim)
    else: 
      self.Label_Label_Energy = PairWiseLoss (self.num_of_labels)

    self.L2 = LabelDeltaL2(self.scale)

  def forward_energy_only (self,input,label,do_print=False):
    # new_label = self.Label_Label_Energy.standard_linear( label )
    ## change to use @new_label ?? new_label
    ## F.sigmoid compress the scalar input, so that very high scalars are all treated equally 
    term1 = self.forward_input_label (input,label)
    if not self.do_label_label : 
      if do_print: 
        print ('term1 Input_Label_Energy')
        print (term1.transpose(0,1))
      return term1 
    else: 
      # term2 = self.Label_Label_Energy.label_compat(new_label) * self.weight_term
      term2 = self.forward_label_label (label)
      if do_print: 
        print ('term1 Input_Label_Energy')
        print (term1.transpose(0,1))
        print ('term2 Label_Label_Energy')
        print (term2.transpose(0,1))
      return (term1 + term2) ## model Energy(x,y)
  
  def forward_input_label (self,input,label): 
    term1 = torch.sum( -1* F.sigmoid(input) * label, dim=1 ) ## @label if we don't wrap a variable here, we will need retain_graph=True 
    return term1.unsqueeze(1) ## make into 2D, size batch_size x 1

  def forward_label_label (self,label) : 
    if not self.pair_wise_only: 
      new_label = self.Label_Label_Energy.standard_linear( label )
      return self.Label_Label_Energy.label_compat(new_label) * self.weight_term
    else: 
      return self.Label_Label_Energy.label_compat(label) * self.weight_term

  def forward_delta_L2 (self,guess_label,true_label):
    if self.scale == 0 : 
      return Variable ( torch.zeros((len(guess_label),1)).cuda() ) ## make into batch_size x 1 
    return self.L2.forward(true_label,guess_label)


class StructureEnergyLoss (nn.Module):
  def __init__(self, feature_nn_model, num_of_labels, label_layer_dim, do_label_label, L2_scale=1, epoch_y=100, epoch_predict_y=100, lr_optim_y=.001, lr_predict_y=.001, wait_until_y=100, epoch_params=1, lr_params=.0001, batch_size=16, do_gcnn=False, init_label_emb=False, load_gcnn_emb=False, not_train_w2v_emb=False, not_update_cnn_att=False, weight_term=1, lr_lagran=.1, pair_wise_only=False ): # true_label,guess_label
    super(StructureEnergyLoss, self).__init__()

    self.not_update_cnn_att = not_update_cnn_att
    self.not_train_w2v_emb = not_train_w2v_emb
    self.load_gcnn_emb = load_gcnn_emb
    self.do_gcnn = do_gcnn 
    self.init_label_emb = init_label_emb

    self.do_label_label = do_label_label # do we need to do label-label energy ?? 

    self.num_of_labels = num_of_labels # retain original num_of_labels
    self.label_layer_dim = label_layer_dim

    self.energy_function = Energy_Function(self.num_of_labels, self.label_layer_dim, do_label_label=self.do_label_label,scale=L2_scale,weight_term=weight_term,pair_wise_only=pair_wise_only) # for both input-label and label-label

    self.label_indexing = Variable(torch.LongTensor ( range(self.num_of_labels) ) )
    self.feature_nn_model = feature_nn_model.cuda()  # much easier to pass self.feature_nn_model so that we can call the model.params

    self.inf = Variable ( torch.Tensor( [float("Inf")] ) ).cuda()  
    self.zero = torch.zeros((1,1)).cuda() 

    self.batch_size = batch_size
    self.UP_BOUND_AT_1 = Variable(torch.ones(self.num_of_labels)).cuda() ## bound the labels to be within [0,1]

    self.epoch_predict_y = epoch_predict_y
    self.epoch_y = epoch_y 
    self.lr_predict_y = lr_predict_y
    self.lr_optim_y = lr_optim_y
    self.lr_lagran = lr_lagran

    self.epoch_params = epoch_params 
    self.lr_params = lr_params

    self.wait_until_y = wait_until_y
    
  def project_01_space (self, x): # x is row x 1 vector Tensor (not Variable)
    # if x is outside the 0-1 space, project it back (cheap approximation)
    x = torch.min( x , self.UP_BOUND_AT_1 ) ## possible we get error in backprop if we use x[x>1]=1 ???
    x = F.relu(x) ## same as doing max(x,0)
    return x

  def turn_on_training (self): 
    self.feature_nn_model.train()
    self.energy_function.train()
    self.energy_function.Label_Label_Energy.train()
  
  def turn_on_eval (self): 
    self.feature_nn_model.eval()
    self.energy_function.eval()
    self.energy_function.Label_Label_Energy.eval()

  def freeze_Encoder_params (self):
    for param in self.feature_nn_model.parameters():
      param.requires_grad = False
    for param in self.energy_function.parameters():
      param.requires_grad = False

  def unfreeze_Encoder_params (self):
    if self.not_update_cnn_att:  
      for param in self.feature_nn_model.parameters():
        param.requires_grad = False  
    else: 
      for param in self.feature_nn_model.parameters():
        param.requires_grad = True

    # what if we want to keep the same gcnn emb ?? 
    # we will have to freeze the w2v emb as well, because w2v emb is linked to the gcnn
    if self.load_gcnn_emb: 
      ## freeze gcnn because it will cost too much time to train ?? 
      for param in self.feature_nn_model.neighbors_layer1.parameters():
        param.requires_grad = False
      for param in self.feature_nn_model.neighbors_layer2.parameters():
        param.requires_grad = False
      ## must free emb as well ?? 
      self.feature_nn_model.embedding.weight.requires_grad = False
  
    # freeze word emb
    if self.not_train_w2v_emb: ## turn off embedding update. notice, we should need both @self.load_gcnn_emb @self.not_train_w2v_emb ??
      self.feature_nn_model.embedding.weight.requires_grad = False
    else: 
      self.feature_nn_model.embedding.weight.requires_grad = True

    for param in self.energy_function.parameters():
      param.requires_grad = True

  def forward_on_y (self, xy_relation, true_label, label_to_be_optim):
    ## when we optim-y, we can ignore the true-label
    E_XYhat = self.energy_function.forward_energy_only( xy_relation, label_to_be_optim )
    delta_y = self.energy_function.forward_delta_L2( true_label, label_to_be_optim )
    batch_energy = E_XYhat - delta_y ## note the sign +/- switches, because we use min-gradient to optimize a max-problem
    # do not take max(0, ...) because we only need to find y-argmax
    return batch_energy.sum() # / len (xy_relation) ## average

  def forward_all_params (self, label_data, seq_tensor, seq_lengths, select_label, guess_label, true_label, no_update=True, new_label_emb=None, do_print=False):

    ## this function is only for simple forward-pass computation
    ## if this funciton gives 0, then the guess_label == true_label
    ## it is a little faster because we can use @self.feature_nn_model in batch mode
    ## @seq_tensor is going to be a batch. batch x word_indexing
    ## @xy_relation is batch x cnn_enc_dim x label

    if no_update:
      self.turn_on_eval() 

    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb ) ## able to handle batch
    E_XY = self.energy_function.forward_energy_only( xy_relation, true_label ) ## energy input to true label
    E_XYhat = self.energy_function.forward_energy_only( xy_relation, guess_label )
    delta_y = self.energy_function.forward_delta_L2( true_label, guess_label )

    batch_energy = delta_y - E_XYhat + E_XY
    batch_energy = F.relu( batch_energy ) ## pointwise max with 0

    return batch_energy.sum() # / len (xy_relation)  ## average

  def energy_new_sample ( self, xy_relation, guess_label ):
    E_XYhat = self.energy_function.forward_energy_only( xy_relation, guess_label )
    return E_XYhat.sum() # / len (xy_relation) ## take average, we will take deriv. wrt guess_label.

  def dual_constraint_loss (self, xy_relation,guess_label, true_label, lambda1, lambda2 ):

    # minimize y (solve the dual)
    delta_y = -1 * self.energy_function.forward_delta_L2( guess_label,true_label ) 
    delta_y = delta_y.sum() 

    ## on true label 
    term1y = self.energy_function.forward_input_label (xy_relation,true_label)
    ## on guess label 
    term1yHat = self.energy_function.forward_input_label (xy_relation,guess_label)
    ## lagrangian 1 
    input_label = lambda1 @ (term1yHat - term1y) 

    ## 
    if self.do_label_label: 
      term2y = self.energy_function.forward_label_label(true_label) 
      term2yHat = self.energy_function.forward_label_label(guess_label) 
      label_label = lambda2 @ (term2yHat - term2y) 
    else: 
      term2y = 0 
      term2yHat = 0 
      label_label = 0 

    loss = delta_y + input_label + label_label 
    return loss , term1yHat , term1y , term2yHat , term2y

  def optim_y_constraint ( self, label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='near_true',new_label_emb=None,do_print=False,lambda1=None,lambda2=None, keep_feasible=False ) :

    num_people = seq_tensor.shape[0]
    init = deepcopy ( true_label )
  
    self.turn_on_eval() ## we need get the best value before we even do any gradient 
    self.freeze_Encoder_params() ## need to remove params that not require gradients
    
    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor,seq_lengths, select_label, new_label_emb=new_label_emb ) # we need the feature. feature is considered fixed when optim-y
  
    best_loss = -self.inf
    last_loss = -self.inf
    primal_best_guess_label = Variable( init.cuda() , requires_grad=False ) 

    ## make the langrangian 
    if lambda1 is None: 
      lambda1 = F.relu(torch.ones( (1,num_people) ) * self.num_of_labels ).cuda()
    if lambda2 is None: 
      lambda2 = F.relu(torch.ones( (1,num_people) ) * self.num_of_labels ).cuda() 

    for outer in range (20): 

      best_energy = self.inf ## reset everything for the y-loop
      last_energy = self.inf
      best_guess_label = Variable( init.cuda() , requires_grad=False ) 

      guess_label = Variable( init.cuda() , requires_grad=True )  ## feasible point
      optimizer = optim.SGD([guess_label], lr = self.lr_optim_y, momentum=0.9)

      for i in range(self.epoch_y):

        optimizer.zero_grad()
        loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,guess_label, true_label, lambda1, lambda2 )
        loss.backward(retain_graph=True) 
        optimizer.step()

        ## update
        guess_label.data = deepcopy(self.project_01_space(guess_label).data) ## fix the y-argmax so it doesn't go out of 0-1 bound
        loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,guess_label, true_label, lambda1, lambda2 )

        if (i % 5 == 0) and do_print: 
          print ('1 batch, outer {} iter {}, lr {}, constraint loss {}'.format(outer,i,optimizer.param_groups[0]['lr'],loss) )
    
        if (i > 5):
          if ( torch.abs( loss - best_energy ) < 10**-6 ) or ( torch.abs( loss - last_energy ) < 10**-6 ) : # break early
            break
        
        last_energy = loss # update to track the trend.
        if loss < best_energy: ## smaller the better
          best_energy = loss
          best_guess_label = guess_label
        
      ## now we handle the lagrangian 
      ## outer: update lagrangian 
      loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,best_guess_label, true_label, lambda1, lambda2 )
      lambda1 = F.relu( lambda1 + self.lr_lagran/(1.0+outer) * (term1yHat-term1y).transpose(0,1) ) ## greater than 0 
      if self.do_label_label: 
        lambda2 = F.relu( lambda2 + self.lr_lagran/(1.0+outer) * (term2yHat-term2y).transpose(0,1) ) 

      primal_loss = self.energy_function.forward_delta_L2( best_guess_label,true_label ) 
      primal_loss = primal_loss.sum() 
      if do_print: 
        print ('primal value {}'.format(primal_loss))
       
      if outer > 2: 
        if ( torch.abs( primal_loss - best_loss ) < 10**-4 ) or ( torch.abs( primal_loss - last_loss ) < 10**-4 )  : # break early
          break
       
      last_loss = primal_loss # update to track the trend.
      if primal_loss >= best_loss: ## update the best_loss 
        best_loss = last_loss 
        primal_best_guess_label = best_guess_label
  
    ## return only feasible ... ?? 
    # primal_best_guess_label.requires_grad = False ## at this point, we don't care about updating this anymore 
    if keep_feasible: ## replace what's not feasible 
      where_feasible = self.is_feasible ( xy_relation, primal_best_guess_label, true_label, return_not=True ) ## not feasible keep as cnn ?? 
      if len(where_feasible) > 0: 
        primal_best_guess_label.data[where_feasible] = true_label.data[where_feasible] ## basically replace where cnn-best can be improved. ??? 


    ## NOTE: we must compute the new energy after projecting the y-argmax
    self.turn_on_eval()
    energy_value = self.forward_all_params( label_data, seq_tensor, seq_lengths, select_label, primal_best_guess_label, true_label, no_update=True, new_label_emb=new_label_emb ) ## do a full complete forward pass.
    ## return the prediction
    if (energy_value>0) :
      return energy_value , primal_best_guess_label ## must reverse the sign because we do min(-value)
    else:
      return Variable(torch.zeros((1))).cuda() , primal_best_guess_label ## negative energy, so return 0 because of max ( ..., 0 )

  def optim_new_sample (self, label_data, seq_tensor, seq_lengths, select_label, new_label_emb, last_best_guess_label=None, do_print=False, near_cnn=0, cnn_best_label=None ):

    self.freeze_Encoder_params() ## need to remove params that not require gradients
  
    num_people = seq_tensor.shape[0]
    if last_best_guess_label is None: 
      guess_label = Variable ( torch.zeros( num_people , self.num_of_labels ).cuda() + 0.5001 , requires_grad=True )  ## init at 0.5 for all predictions
    else: 
      guess_label = Variable ( torch.FloatTensor(last_best_guess_label).cuda(), requires_grad=True )
    
    if (near_cnn > 0) : ## constraint to not move far away @last_best_guess_label, here we want @last_best_guess_label to be @cnn_best_label 
      cnn_best_label = Variable ( torch.FloatTensor(cnn_best_label).cuda(), requires_grad=False )
      
    self.turn_on_eval() 
    # input_sequence_emb, sent_len, label_emb, new_label_emb
    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb ) ## able to handle batch  ## feature
    best_energy = self.energy_new_sample ( xy_relation , guess_label )
    best_guess_label = deepcopy ( guess_label )
    last_energy = self.inf
    
    optimizer = optim.SGD([guess_label], lr = self.lr_predict_y, momentum=0.9 ) # SGD 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    for i in range (self.epoch_predict_y):

      scheduler.step(i)

      optimizer.zero_grad()
      E_XYhat = self.energy_new_sample ( xy_relation , guess_label )
      if near_cnn > 0 : ## constraint 
        diff_cnn = self.energy_function.forward_delta_L2( guess_label, cnn_best_label )
        diff_cnn = diff_cnn.sum() 
        E_XYhat = E_XYhat + near_cnn*diff_cnn
        
      E_XYhat.backward( retain_graph=True )
      optimizer.step() ## this automatically update @guess_label

      ## update
      guess_label.data = deepcopy(self.project_01_space(guess_label).data) ## fix the y-argmax so it doesn't go out of 0-1 bound
      this_energy = self.energy_new_sample ( xy_relation, guess_label ) ## need to compute the energy again, because after we project, we do NOT get the same energy value.

      if (i % 10 ==0) and do_print:
        print ('energy E_XYhat for new sample iter {}, {}, lr {}'.format(i,this_energy,optimizer.param_groups[0]['lr']) ) 

      if (i > self.wait_until_y):
        if ( torch.abs( this_energy - best_energy ) < 10**-6 ) or ( torch.abs( this_energy - last_energy ) < 10**-6 ) : # break early
          break

      last_energy = this_energy # update to track the trend.

      if this_energy < best_energy: ## smaller the better
        best_energy = this_energy
        best_guess_label = guess_label
      
    return best_guess_label

  def optim_new_sample_constraint (self, label_data, seq_tensor, seq_lengths, select_label, new_label_emb, last_best_guess_label=None, do_print=False, near_cnn=0, cnn_best_label=None, lambda1=None, lambda2=None, keep_feasible=False ):

    ## use lagrangian to force both input-label and label-label energies to go down. 
    
    num_people = seq_tensor.shape[0]
    init = torch.FloatTensor(cnn_best_label)

    self.turn_on_eval() 
    self.freeze_Encoder_params() ## need to remove params that not require gradients
  
    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb ) ## able to handle batch  ## feature
 
    best_loss = -self.inf
    last_loss = -self.inf
    primal_best_guess_label = Variable( init.cuda() , requires_grad=False ) 

    cnn_best_label = Variable ( torch.FloatTensor(cnn_best_label).cuda(), requires_grad=False )

    ## make the langrangian 
    if lambda1 is None: 
      lambda1 = F.relu(torch.ones( (1,num_people) ) * self.num_of_labels ).cuda()
    if lambda2 is None: 
      lambda2 = F.relu(torch.ones( (1,num_people) ) * self.num_of_labels ).cuda() 

    for outer in range (20): 

      best_energy = self.inf ## reset everything for the y-loop
      last_energy = self.inf
      best_guess_label = Variable( init.cuda() , requires_grad=False ) 

      guess_label = Variable ( init.cuda(), requires_grad=True ) ## feasible ? 
      optimizer = optim.SGD([guess_label], lr = self.lr_predict_y, momentum=0.9 ) # SGD 

      for i in range (self.epoch_predict_y):

        optimizer.zero_grad()
        loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,guess_label, cnn_best_label, lambda1, lambda2 )
        loss.backward(retain_graph=True) 
        optimizer.step()

        ## update
        guess_label.data = deepcopy(self.project_01_space(guess_label).data) ## fix the y-argmax so it doesn't go out of 0-1 bound
        loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,guess_label, cnn_best_label, lambda1, lambda2 )

        if (i % 15 == 0) and do_print: 
          print ('1 batch, outer {} iter {}, lr {}, constraint loss {}'.format(outer,i,optimizer.param_groups[0]['lr'],loss) )
         
        if (i > 5):
          if ( torch.abs( loss - best_energy ) < 10**-6 ) or ( torch.abs( loss - last_energy ) < 10**-6 ) : # break early
            break
        
        last_energy = loss # update to track the trend.
        if loss < best_energy: ## smaller the better
          best_energy = loss
          best_guess_label = guess_label

      ## now we handle the lagrangian 
      ## outer: update lagrangian 
      loss , term1yHat , term1y , term2yHat , term2y = self.dual_constraint_loss (xy_relation,best_guess_label, cnn_best_label, lambda1, lambda2 )
      lambda1 = F.relu( lambda1 + self.lr_lagran/(1.0+outer) * (term1yHat-term1y).transpose(0,1) ) ## greater than 0 
      if self.do_label_label: 
        lambda2 = F.relu( lambda2 + self.lr_lagran/(1.0+outer) * (term2yHat-term2y).transpose(0,1) ) 

      primal_loss = self.energy_function.forward_delta_L2( best_guess_label,cnn_best_label ) 
      primal_loss = primal_loss.sum() 
      if do_print: 
        print ('primal value {}'.format(primal_loss))
   
      if outer > 2: 
        if ( torch.abs( primal_loss - best_loss ) < 10**-4 ) or ( torch.abs( primal_loss - last_loss ) < 10**-4 )  : # break early
          break
       
      last_loss = primal_loss # update to track the trend.
      if primal_loss >= best_loss: ## update the best_loss 
        best_loss = last_loss 
        primal_best_guess_label = best_guess_label
      
    ## return only feasible ... ?? 
    if keep_feasible: 
      where_feasible = self.is_feasible ( xy_relation, primal_best_guess_label, cnn_best_label, return_not=True ) ## not feasible keep as cnn ?? 
      if len(where_feasible) > 0: 
        primal_best_guess_label.data[where_feasible] = cnn_best_label.data[where_feasible] ## basically replace where cnn-best can be improved. ??? 

    return primal_best_guess_label 

  def is_feasible (self, this_xy, guess_label, true_label, extract_first_entry=False, return_not=False ) : ## both energy decreases 
    
    # extract_first_entry is meant for brute force method
    # this_xy is already "expanded" to have the same matrix size as @guess_label which is num_choices x num_label 

    if extract_first_entry: 
      term1_true = self.energy_function.forward_input_label (this_xy[0].unsqueeze(0), true_label) # term1_true is column vector 
    else: 
      term1_true = self.energy_function.forward_input_label (this_xy, true_label) # term1_true is column vector

    term1_guess = self.energy_function.forward_input_label (this_xy,guess_label)
    feasible = term1_guess <= term1_true

    if self.do_label_label: 
      term2_true = self.energy_function.forward_label_label (true_label)
      term2_guess = self.energy_function.forward_label_label (guess_label)
      feasible_term2 = term2_guess <= term2_true
      feasible = feasible * feasible_term2 ## both 1 

    ## want to avoid transfering to cpu as much as possible, so don't want to conver to numpy 
    if return_not: 
      return torch.nonzero(feasible==0)[:,0] ## because col vector 
    else: 
      return torch.nonzero(feasible==1)[:,0] ## because col vector 

    # feasible = feasible.cpu().data.numpy()
    # return np.where(feasible==1)[0] ## because col vector 

  def optim_new_sample_brute_force (self, label_data, seq_tensor, seq_lengths, select_label, new_label_emb, last_best_guess_label=None, do_print=False, near_cnn=0, cnn_best_label=None, true_label=None, permutation_choice=None ) : 
    
    self.freeze_Encoder_params() ## need to remove params that not require gradients
    self.turn_on_eval() 

    guess_label = Variable ( torch.FloatTensor ( permutation_choice[self.num_of_labels] ).cuda() , requires_grad=False )

    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor , seq_lengths, select_label, new_label_emb ) ## able to handle batch  ## feature

    best_guess_label = Variable ( torch.zeros( seq_tensor.shape[0] , self.num_of_labels ).cuda() , requires_grad=False )  ## init at 0.5 for all predictions
 
    for ob in range (seq_tensor.shape[0]): ## for each ob, we get the best brute force choice. 
      this_xy = xy_relation[ob].expand(guess_label.shape[0], xy_relation.shape[1]) ## same dim as guess label. 2D, # batch x label
      best_energy = self.energy_function.forward_energy_only( this_xy, guess_label )
      
      if true_label is not None: 
        where_feasible = self.is_feasible ( this_xy, guess_label, true_label[ob].unsqueeze(0), extract_first_entry=True, return_not=False )

        if len(where_feasible) > 0 : 
          best_energy = best_energy[where_feasible] ## select feasible row, for the energy
          feasible_label = guess_label[where_feasible] ## select the feasible labels for these feasible energy 
          max_val, max_ind = torch.min( best_energy , dim=0 ) ## results is 1 x batch_size, here, batch_size is the len of @guess_label 
          best_guess_label[ob] = feasible_label[max_ind] ## keep the best guess label for this obs. 
        else: 
          max_val, max_ind = torch.min( best_energy , dim=0 ) ## results is 1 x batch_size, here, batch_size is the len of @guess_label 
          best_guess_label[ob] = guess_label[max_ind] ## keep the best guess label for this obs. 
      
      ## true_label not exists. (this is for testing phase)    
      else: 
        max_val, max_ind = torch.min( best_energy , dim=0 ) ## results is 1 x batch_size, here, batch_size is the len of @guess_label 
        best_guess_label[ob] = guess_label[max_ind] ## keep the best guess label for this obs. 
   
    return best_guess_label

  def optim_params (self, label_data, seq_tensor,seq_lengths,true_label, select_label, guess_label, new_label_emb=None, do_print=False):

    ## record current value. because iter 0 may update to the higher output 
    self.turn_on_eval() 
    best_energy = self.forward_all_params ( label_data, seq_tensor, seq_lengths, select_label, guess_label, true_label, no_update=True, new_label_emb=new_label_emb) ## we don't need to recompute @delta_y
    best_state_dict = {'model_state_dict': deepcopy (self.state_dict()) } 
    if do_print: 
      print ('\nbefore optim params {}'.format(best_energy))

    self.turn_on_training() 
    self.unfreeze_Encoder_params() ## turn all other parameters on.
  
    optimizer = optim.SGD( filter(lambda p: p.requires_grad, self.parameters()), lr = self.lr_params, momentum=0.9 ) ## need to remove params that not require gradients
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    for i in range (self.epoch_params): ## we can do epoch_params>1, but the theory says that we optim y, then do 1-update for params 
      
      scheduler.step(i)

      optimizer.zero_grad() # compute forward, then get gradient
      energy = self.forward_all_params ( label_data, seq_tensor, seq_lengths, select_label, guess_label, true_label, no_update=False, new_label_emb=new_label_emb) ## we don't need to recompute @delta_y
      energy.backward( retain_graph=True )
      optimizer.step()

      if do_print and (i%5==0): 
        print ('during optim params see func val after iter {} , {}'.format(i, energy) )

      if energy < best_energy : ## only use this if @epoch_params > 1
        best_energy = energy
        ## notice, if we call @backward and @step each iter, then we will end up having param that are not "best"
        best_state_dict = {'model_state_dict': deepcopy (self.state_dict()) } #'optimizer_state_dict': optimizer.state_dict()}
        if do_print:
          print ('save state_dict, iter {} energy {}'.format(i,energy) )
        break ## we don't need to overdo and find the "really best", these params will be updated for the next batch anyway 
       
      if best_energy == 0: 
        break
          
    ## reset to best_state_dict 
    self.load_state_dict( best_state_dict['model_state_dict'] )
    self.turn_on_eval() 
    energy = self.forward_all_params ( label_data, seq_tensor, seq_lengths, select_label, guess_label, true_label, no_update=True, new_label_emb=new_label_emb) ## we don't need to recompute @delta_y
    if do_print: 
      print ('see last energy which should be the best energy {}'.format(energy) )

    return energy ## variable ? tensor ? 

  def optim_y ( self, label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='near_true',new_label_emb=None,do_print=False): # this is done in batch mode. We find argmax-y for each sample SIMULTANEOUSLY

    ## 
    num_people = seq_tensor.shape[0]
    init =  0.9999 * true_label + 0.00001 
    if start_option=='uniform': 
      # better option ?
      init = torch.zeros( (num_people, self.num_of_labels) ) + 0.5001  ## init at 0.5 for all predictions
    if start_option=='random': 
      # random 
      init = F.sigmoid ( torch.randn( num_people, self.num_of_labels ) )
    if start_option=='near_zero': 
      init = torch.zeros( (num_people, self.num_of_labels) ) + 0.0001  ## init at 0.5 for all predictions

  
    self.turn_on_eval() ## we need get the best value before we even do any gradient 

    guess_label = Variable( init.cuda() , requires_grad=True )  
    xy_relation = self.feature_nn_model.do_forward( label_data, seq_tensor,seq_lengths, select_label, new_label_emb=new_label_emb ) # we need the feature. feature is considered fixed when optim-y
    best_energy = self.forward_on_y( xy_relation, true_label, guess_label )
    best_guess_label = Variable( init.cuda() , requires_grad=False ) 
    last_energy = self.inf

    self.freeze_Encoder_params() ## need to remove params that not require gradients
    optimizer = optim.SGD([guess_label], lr = self.lr_optim_y, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    for i in range(self.epoch_y):

      scheduler.step(i)

      optimizer.zero_grad()
      energy_value = self.forward_on_y( xy_relation, true_label, guess_label )
      energy_value.backward( retain_graph=True ) # retain_graph=True
      optimizer.step()

      ## update
      guess_label.data = deepcopy (self.project_01_space(guess_label).data)  ## fix the y-argmax so it doesn't go out of 0-1 bound
      this_energy = self.forward_on_y( xy_relation, true_label, guess_label ) ## after we project @guess_label, we need to AGAIN compute @guess_label

      if (i % 10 == 0) and do_print: 
        print ('1 batch after argmax y, min-negative so value should decrease, iter {} , {}, lr {}'.format(i,this_energy, optimizer.param_groups[0]['lr'] ))
   
      if (i > self.wait_until_y):
        if ( torch.abs( this_energy - best_energy ) < 10**-6 ) or ( torch.abs( this_energy - last_energy ) < 10**-6 ) : # break early
          break

      last_energy = this_energy ## update previous value 

      ## energy DECREASES because we're minimizing "NEGATIVE" OF F(y), which same as maximizing F(y)
      if this_energy < best_energy: ## notice, we record small energy value because we do arg-min for "negative y"
        best_energy = this_energy # update best energy value
        best_guess_label = guess_label
   
    # END LOOP
    ## NOTE: we must compute the new energy after projecting the y-argmax
    self.turn_on_eval()
    energy_value = self.forward_all_params( label_data, seq_tensor, seq_lengths, select_label, best_guess_label, true_label, no_update=True, new_label_emb=new_label_emb ) ## do a full complete forward pass.
    if do_print : 
      print ('full energy with params {}'.format(energy_value) ) 

    ## return the prediction
    if (energy_value>0) :
      return energy_value , best_guess_label ## must reverse the sign because we do min(-value)
    else:
      return Variable(torch.zeros((1))).cuda() , best_guess_label ## negative energy, so return 0 because of max ( ..., 0 )

  def train_1_batch (self, label_data, seq_tensor,seq_lengths,true_label, select_label, adjacency_parent=None, adjacency_children=None, new_label_emb=None, do_print=False, constraint=False, keep_feasible=False ) : ## at batch b 
    ## return energy for 1 batch. this requires doing the optim 

    self.turn_on_training() ## must turn on the train-mode. 

    ## call the label layer 
    ## if init label then we DO NOT need to update @self.label because there is an internal layer being treated as the labels 
    ## if we call @self.load_gcnn_emb, then we do not update emb, and so not update label emb and not update gcnn layer
    if (self.do_gcnn or (not self.init_label_emb) ) and (not self.load_gcnn_emb )  : 
      self.feature_nn_model.do_forward_label(label_data, adjacency_parent, adjacency_children, new_label_emb) ## create the labels to be used once for all forward step. ... save time. 
    
    # print ('see')
    # print (self.feature_nn_model.new_label_emb)

    # for 1 batch, we iterate over each batch, maximize their argmax-y
    # then, we find the best params ( the params update will only "be active" wrt this batch... is this the best choice? )

    ## optim the argmax-y for 1 batch (this is about 16 or 32 or 64 samples simultaneously)

    if not constraint: 
      energy_value, new_pred_label = self.optim_y ( label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='near_true', new_label_emb=self.feature_nn_model.new_label_emb, do_print=do_print )
    
    else: 
      energy_value, new_pred_label = self.optim_y_constraint ( label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='near_true', new_label_emb=self.feature_nn_model.new_label_emb, do_print=do_print, keep_feasible=keep_feasible )
    
    # if energy_value <= 0: # ONLY UPDATE PARAMS IF max(...,0) give "non-zero", but we want to update the params because we will never truly find the best y. 
    #   ## !! we start near true label so that we can get some "increase" energy 
    #   print ('restart at near_true')
    #   energy_value, new_pred_label = self.optim_y (  label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='near_true', new_label_emb=new_label_emb, do_print=do_print)
    # if energy_value <= 0: 
    #   ## !! we can get some "increase" energy ??
    #   print ('restart at random')
    #   energy_value, new_pred_label = self.optim_y (  label_data, seq_tensor,seq_lengths,true_label, select_label, start_option='random', new_label_emb=new_label_emb, do_print=do_print)

    # now we do optim w.r.t. the params 
    if energy_value <= 0: # ONLY UPDATE PARAMS IF max(...,0) give "non-zero"
      if do_print: 
        print ('ONLY UPDATE PARAMS IF max(...,0) give non-zero')
      return energy_value
    else: 
      energy_value = self.optim_params ( label_data, seq_tensor,seq_lengths,true_label, select_label, new_pred_label, new_label_emb=self.feature_nn_model.new_label_emb, do_print=do_print )
      return energy_value ## only need to return @energy_value, this is the final energy after operation: min_params (max_y function ) FOR 1 BATCH

  # def test_1_batch (self, label_data, seq_tensor,seq_lengths, select_label ): 
  #   best_guess_label = self.optim_new_sample (label_data, seq_tensor, seq_lengths, select_label)
  #   return best_guess_label.cpu().data.numpy()
