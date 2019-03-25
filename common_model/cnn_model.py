
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import xavier_uniform_


class EncoderCNN ( nn.Module ): 
	def __init__(self, num_of_word, word_emb_dim, num_of_filter, width_of_filter ): # @num_of_word is vocab size. 
		super().__init__()
		## note... batch_size doesn't matter, by default the batch_size can be handled by using 3d tensor. 
		self.word_emb_dim = word_emb_dim # @word_emb_dim is emb size (i.e. 300 in GloVe)
		# self.embedding = nn.Embedding(num_of_word, word_emb_dim) ## @embedding, row=word, col=some dimension value 
		if (width_of_filter % 2 == 0): ## even number
			print ('even value for left/right padding will not work, so we add 1 to make it odd.')
			width_of_filter = width_of_filter+1 ## if @width_of_filter is even, then we don't have equal padding on both side 
		padding_num = (width_of_filter - 1)//2 
		self.dropout1 = nn.Dropout(p=0.2)

		self.conv1d = nn.Conv1d(word_emb_dim, num_of_filter, width_of_filter, stride=1, padding=padding_num, bias=True ) # @num_of_filter is the dim of the emb-vector after the CNN
		xavier_uniform_(self.conv1d.weight)

	def forward(self, embeds): ## this is to be used with another neural-layer, so we don't give it its own embed. 
		# MAKE SURE TO HAVE DUMMY PADDING. 
		## note... @seq_lengths doesn't matter, we already do the len-padding in preprocessing
		# embeds = self.embedding(sentence) ## @embeds dim is batch_size x word_emb_dim x sent_len 
		embeds = self.dropout1( embeds )
		# conv_out = F.tanh( self.conv1d( embeds.transpose(1,2) ) ) ## non-linear activation 
		conv_out = self.conv1d( embeds.transpose(1,2) )
		return conv_out ## batch_size x conv_emb_dim x sent_len 


class LabelEncoderCNN ( nn.Module ): 
	def __init__(self, num_of_word, word_emb_dim, num_of_filter, width_of_filter ): # @num_of_word is vocab size. 
		super().__init__()
		## note... batch_size doesn't matter, by default the batch_size can be handled by using 3d tensor. 
		self.word_emb_dim = word_emb_dim # @word_emb_dim is emb size (i.e. 300 in GloVe)
		# self.embedding = nn.Embedding(num_of_word, word_emb_dim) ## @embedding, row=word, col=some dimension value 
		if (width_of_filter % 2 == 0): ## even number
			print ('even value for left/right padding will not work, so we add 1 to make it odd.')
			width_of_filter = width_of_filter+1 ## if @width_of_filter is even, then we don't have equal padding on both side 
		padding_num = (width_of_filter - 1)//2 
		self.dropout1 = nn.Dropout(p=0.2)

		self.conv1d = nn.Conv1d(word_emb_dim, num_of_filter, width_of_filter, stride=1, padding=padding_num ) # @num_of_filter is the dim of the emb-vector after the CNN
		xavier_uniform_(self.conv1d.weight)

	def forward(self, embeds): ## this is to be used with another neural-layer, so we don't give it its own embed. 
		# MAKE SURE TO HAVE DUMMY PADDING. 
		## note... @seq_lengths doesn't matter, we already do the len-padding in preprocessing
		# embeds = self.embedding(sentence) ## @embeds dim is batch_size x word_emb_dim x sent_len 
		embeds = self.dropout1( embeds )
		# conv_out = F.tanh( self.conv1d( embeds.transpose(1,2) ) ) ## non-linear activation 
		conv_out = self.conv1d( embeds.transpose(1,2) ) 
		return conv_out.transpose(1,2) ## batch_size x num(words) x conv_emb_dim


class LabelEncoderAve ( nn.Module ): ## take average, pass through linear. ... faster ?? 
	def __init__(self, num_of_word, word_dim_in ): # @num_of_word is vocab size. 
		super().__init__()
		## note... batch_size doesn't matter, by default the batch_size can be handled by using 3d tensor. 
		# self.linear_sequential = nn.Sequential(	nn.Dropout(p=0.1),
    #                                      		nn.Linear(word_dim_in, word_dim_out), nn.Tanh() ) 

	def forward(self, embeds, label_len): ## this is to be used with another neural-layer, so we don't give it its own embed. 
		# MAKE SURE TO HAVE DUMMY PADDING. 
		# @label_len must be 2D num_obs x 1 
		## note... @seq_lengths doesn't matter, we already do the len-padding in preprocessing
		# embeds = self.embedding(sentence) ## @embeds dim is batch_size x word_emb_dim x sent_len
		label_len = label_len.type(torch.FloatTensor).cuda() ## lengths were kept as LongTensor, and not sent to cuda
		label_len = label_len.view(len(label_len),1) ## convert len from torch.LongTensor(label_len) into 2D vectors 
		emb_ave_over_word = torch.sum(embeds, 1) / label_len ## take average of words in the description, this broadcast the len division 
		return self.linear_sequential ( emb_ave_over_word ) ## batch_size x 1 x conv_emb_dim


class LogisticsWeight (nn.Module):
	def __init__(self,num_of_labels,hidden_dim): # @hidden_dim is dim of output after the CNN layer. 
		super(LogisticsWeight, self).__init__()
		self.num_of_labels = num_of_labels ## all the label in the whole data 
		self.hidden_dim = hidden_dim
		self.B = nn.Parameter(torch.randn(num_of_labels, 1, hidden_dim)) # @self.B is numDiagnosis x 1 x hiddenDim, hence, B[1] is vector fo label 1 
		xavier_uniform_(self.B)
		self.C = nn.Parameter(torch.randn(1,num_of_labels)) # constant aka bias term 
		xavier_uniform_(self.C)

	def forward(self, note_embedding): ## note, at this stage, we do NOT do batch-mode 
		# @note_embedding is num_of_labels x hidden_state_dim
		# note, for one trait ell, we need B_ell^t V_ell, this is the same as diag ( B x V ), 
		# but instead of "diag" we will use batch-multiplication to mult. the corresponding vectors
		# note: torch.view is not the same as torch.transpose , view will fills by row based on the new size.
		note_embedding = note_embedding.view(self.num_of_labels,self.hidden_dim,1) ## turn into 3D tensor 
		prediction = torch.bmm( self.B, note_embedding ).view(1,self.num_of_labels) ## 2D tensor 
		return prediction + self.C ## ?? not return probability, return the linear regression score, so we can pass into the energy structure 


class cnn_att_model (nn.Module):  # !! TAKE 1.6 SEC PER BATCH ... SLOW !!
	def __init__(self, num_of_labels, num_of_word, word_emb_dim, num_of_filter, width_of_filter, new_label_dim, do_gcnn, init_label_emb):  
		# @new_label_dim is dim for @self.reduce_dim
 
		super(cnn_att_model, self).__init__()
		
		self.do_gcnn = do_gcnn
		self.init_label_emb = init_label_emb

		self.num_of_labels = num_of_labels
		self.num_of_filter = num_of_filter

		self.encoder = EncoderCNN(num_of_word, word_emb_dim, self.num_of_filter, width_of_filter)
		
		## convert feature into same dim as label-dim
		if self.do_gcnn == True: 
			self.reduce_dim = nn.Sequential ( nn.Linear( self.num_of_filter, new_label_dim ) , nn.ReLU() ) 

		if self.do_gcnn == False: 

      ## attention layer @self.regression is a trick, where we use the values as "initalization for ICD9"
			if self.init_label_emb == True:
				self.regression = nn.Linear(self.num_of_filter , self.num_of_labels, bias=False)
				xavier_uniform_(self.regression.weight)

      ## linear regression predictor   
			self.final = nn.Linear(self.num_of_filter , self.num_of_labels)
			xavier_uniform_(self.final.weight)

		self.nn_loss_function = nn.BCEWithLogitsLoss()

	def do_note_embedding(self, cnn_out, sent_len, label): 
		note_embedding = Variable( torch.zeros(cnn_out.shape[0], self.num_of_labels, self.num_of_filter) ).cuda() ## @cnn_out.shape[0] because not every batch has the same size.
		## for each doc in the batch.
		for b in range(cnn_out.shape[0]):
			doc_b = cnn_out[b,:,0:sent_len[b]] ## cnn_out[b] is hidden_state_dim x number_word
			attention = torch.matmul(label,doc_b) # num_of_labels x number_word  
			attention = F.softmax ( attention , dim=1) # num_of_labels x number_word  ## we retain the 3D tensor 
			note_embedding[b] = torch.matmul( attention, doc_b.transpose(0,1)  ) #  num_of_labels x hidden_state_dim
		return note_embedding

	def forward(self, input_sequence_emb, sent_len, label_emb, new_label_emb):

		cnn_out = F.tanh ( self.encoder.forward(input_sequence_emb) ) 
	
		if self.do_gcnn: 
			cnn_out = self.do_note_embedding(cnn_out, sent_len, label_emb)
			label_emb = torch.cat ( ( label_emb, new_label_emb ), dim=1)
			cnn_out = self.reduce_dim(cnn_out)
			prob = torch.sum( label_emb * cnn_out ,dim=2) 
		else:
			if self.init_label_emb == True:				
				cnn_out = self.do_note_embedding(cnn_out, sent_len, self.regression.weight)
			else: 
        ## notice, we must not project back onto @label_emb, it may not make sense because we have a weighted ave. wrt @label_emb projected back onto @label_emb
				cnn_out = self.do_note_embedding(cnn_out, sent_len, label_emb)
			
			prob = self.final.weight.mul(cnn_out).sum(dim=2).add(self.final.bias)
		
		return prob

	def loss_function(self, input_sequence_emb, sent_len, label_emb, true_label, neighbor_vec):
		prediction = self.forward(input_sequence_emb, sent_len, label_emb, neighbor_vec)
		return self.nn_loss_function(prediction, true_label)


class cnn_att_nei_model (cnn_att_model):  

	def __init__(self, num_of_labels, num_of_word, word_emb_dim, num_of_filter, width_of_filter, new_label_dim, do_gcnn, init_label_emb):  
		# @new_label_dim is dim for @self.reduce_dim
		cnn_att_model.__init__( self, num_of_labels, num_of_word, word_emb_dim, num_of_filter, width_of_filter, new_label_dim, do_gcnn, init_label_emb )

		self.embedding = nn.Embedding ( num_of_word, self.init_label_emb )

	def forward (self, input_sequence_emb, sent_len, label_emb, new_label_emb): 
		input_sequence_emb = self.embedding ( input_sequence_emb )
		return cnn_att_model.forward(self,input_sequence_emb, sent_len, label_emb, new_label_emb)
	
	def loss_function(self, input_sequence_emb, sent_len, label_emb, true_label, neighbor_vec):
		input_sequence_emb = self.embedding ( input_sequence_emb )
		return cnn_att_model.loss_function(self, input_sequence_emb, sent_len, label_emb, true_label, neighbor_vec)
