

## already preprocessed by baseline, we only need to read in the vocab.

import pickle , os 
import pandas as pd 
import sys,re


try: 
  os.mkdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')
except: 
  pass 

os.chdir('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

data_in = pd.read_csv( '/u/flashscratch/d/datduong/MIMIC3database/vocab.csv', header=None, dtype=str)

vocab_index_map = {'PADDING':0,'UNKNOWN':1} 
reverse_map = {0:'PADDING',1:'UNKNOWN'}

vocab = list(data_in[0]) 

# do we need to sort ?? let's not do it # vocab.sort() 

for num,val in enumerate ( vocab ):
  vocab_index_map[val] = num + 2 ## because we will have padding and unknown 
  reverse_map[num+2] = val 


vocab_index_map = pickle.dump(vocab_index_map,open('vocab_index_map.pickle','wb'))
reverse_map = pickle.dump(reverse_map,open('vocab_reverse_map.pickle','wb'))


