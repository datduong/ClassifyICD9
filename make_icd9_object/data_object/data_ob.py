
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

class Data (object): 

	def __init__(self,data_text,num_of_label,word_index_map,label_index_map, index_subset=None):

		self.SPECIAL_MAP_ICD = {'719.70':'719.7'} # {'11.8':'011.8', '23.9':'023.9', '719.70':'719.7', '17.0':'17'}
		self.num_of_label = num_of_label  
		self.word_index_map = word_index_map
		self.label_index_map = label_index_map
		self.each_person_visit = {} ## track the hospital visit id for each person 

		self.note_for_obs, self.label_for_obs, self.label_onehot_for_obs = self.read_from_format_text(data_text,index_subset)  

	def read_from_format_text (self,text_file, index_subset=None) : # @text_file must be formatted already 
		df = pd.read_csv (text_file,dtype=str) # 'disch_test_split.csv'
		df['LABELS'] = df['LABELS'].fillna('null')

		note_for_obs = {}
		label_for_obs = {} 
		label_onehot_for_obs = {} 
		row_iterator = df.iterrows() ## is it faster ?? 
		for i, row in row_iterator:

			key = row['SUBJECT_ID']+ " " + row['HADM_ID']
			if row['SUBJECT_ID'] not in self.each_person_visit : 
				self.each_person_visit [row['SUBJECT_ID'] ] = [ row['HADM_ID'] ] 
			else: 
				self.each_person_visit [row['SUBJECT_ID'] ].append ( row['HADM_ID'] )

			# text = row['TEXT'] ## may be a whole lot faster to processed this text 
			note_for_obs[key]  = self.make_word_to_indexing ( row['TEXT'] )
			# label
			label = row['LABELS']
			
			label = list(set(label.split(";"))) ## get unique, because there are duplicated ... why ??

			for l in label: 
				if l in self.SPECIAL_MAP_ICD: 
					label [ label.index(l) ] = self.SPECIAL_MAP_ICD[l] 

			label_for_obs[key] = self.make_label_to_indexing_for_obs( label ) ## labels not have description from bioportal will be removed. 

			onehot = np.zeros([self.num_of_label]) # make into 1-hot 
			onehot [ label_for_obs[key] ] = 1.0 
	
			## subset down to only leaf node ?? 
			if index_subset is not None: 
				onehot = onehot[ index_subset ]

			label_onehot_for_obs[key] =  Variable( torch.FloatTensor(onehot) )

		# return data in dictionary format 
		return note_for_obs, label_for_obs, label_onehot_for_obs

	def make_word_to_indexing (self, input_string): # @word_index_map contains dictionary for each word {word1:index1 ...} 
		input_string = input_string.split() ## split by spacing 
		## truncate ?? 
		del input_string[0:5]
		if len(input_string) > 4000: 
			input_string = input_string[0:4000]
		indexing = []  
		for s in input_string: # for each word in the string 
			if s in self.word_index_map : 
				indexing.append ( self.word_index_map [s] ) 
			else: 
				indexing.append ( 1 ) ## unknown index is 1 

		return Variable ( torch.LongTensor(indexing) ) # want to create output {person1: [wordix1 wordix2 ...] ... }

	def make_label_to_indexing_for_obs (self, label_arr): # @make_label_to_indexing_for_obs is like @make_word_to_indexing
		indexing = [ self.label_index_map[w] for w in label_arr if w in self.label_index_map ] ## remove "null" because it is used as dummy for "no disease"	
		return Variable (torch.LongTensor(indexing) ) ## want { observation:[indexing of labels] }

