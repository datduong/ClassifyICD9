
import pickle , os , sys, re
import pandas as pd 
import numpy as np 

## we have the vocab list 
try: 
  os.mkdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')
except: 
  pass 

  
os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

def submitJobs ( w2v_path ) : 

  print ("must use vocab of notes + vocab of icd")
  vocab_list = pd.read_csv("vocab+icd_index_map.txt",header=None) ## must use vocab of notes + vocab of icd 

  vocab_index_map = {'PADDING':0,'UNKNOWN':1} 
  reverse_map = {0:'PADDING',1:'UNKNOWN'}

  for i in range ( vocab_list.shape[0] ) : 
    vocab_index_map [ vocab_list.loc[i,0] ] = i+2 
    reverse_map [ i+2 ] = vocab_list.loc[i,0]

  #
  pickle.dump(vocab_index_map,open('vocab+icd_index_map_add_padding.pickle','wb'))
  pickle.dump(reverse_map,open('vocab_reverse_map_add_padding.pickle','wb'))

  ## create emb. in numpy 

  word_dim = 300
  pretrain_emb = np.zeros ( (len(vocab_index_map), word_dim ) ) 

  fin = open ("/u/flashscratch/d/datduong/"+w2v_path+".txt",'r') # w2vModel1Gram9Jan2019/w2vModel1Gram9Jan2019.txt
  counter = 0 
  for line in fin: 
    if counter == 0 : 
      counter = 1 # skip line 1
      continue
    line = line.strip().split() 
    word = line[0]
    if word in vocab_index_map:
      pretrain_emb [ vocab_index_map[word] ] = np.array ( line[1:len(line)] )


  pickle.dump(pretrain_emb,"pubmed_go_data_vocab_indexing_prune_pretrain_emb.pickle") ## later used to load into the emb

  print ('num of words and dim ')
  print (pretrain_emb.shape)

  print ('sample')
  print (pretrain_emb[0:5])


if len(sys.argv)<1: ## run script 
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )
   
	
