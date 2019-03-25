
import pickle , os 
import pandas as pd 
import numpy as np 
import sys,re

# from nltk.tokenize import RegexpTokenizer
# #retain only alphanumeric
# tokenizer = RegexpTokenizer(r'\w+')


def Punctuation(string): 
  # punctuation marks 
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  # traverse the given string and if any punctuation 
  # marks occur replace it with null 
  for x in string.lower(): 
    if x in punctuations: 
      string = string.replace(x, " ") 
  # Print string without punctuation 
  return string


stop_word = ['the','of','for','by']

try: 
  os.mkdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')
except: 
  pass 


os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')


# @vocab_index_map HAS THE PADDING ALREADY 
vocab_index_map = pickle.load(open('vocab_index_map.pickle','rb')) ## original vocab from mimic, will not have all the words of icd near root 
reverse_map = pickle.load(open('vocab_reverse_map.pickle','rb'))


## prefered_label is the complete list, has parent terms that are not used, but still link go terms together 
print ('load the labels downloaded from bioportal')
Preferred_Label = pickle.load ( open("Preferred_Label.pickle","rb") ) 


## the order of icd9 for those that are used in the data. 
## !! want to index 1 2 3 for all the icd used in the data. 
## icd not used in data (parents terms) are appended to the end. 
## this approach will make the extraction of icd9 and fitting icd much easier. 

icd_in_data = pd.read_csv( '/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_PROCEDURES_ICD_dot_format_to_leaf.csv' , dtype=str, sep=',' )
icd_in_data = icd_in_data.dropna()
icd_in_data = list ( set ( icd_in_data['ICD9_CODE'] ) ) 
icd_in_data.sort() 

##
icd_description = pd.read_csv ( '/u/flashscratch/d/datduong/MIMIC3database/ICD9_descriptions_regex_format.csv', dtype=str,sep="," )
icd_description = icd_description.sort_values(by="ICD9_CODE") ## sort so that we can get the same matching order as @icd_in_data 

print ('num in data but not in description file, should be zero ?? ')
z = [ i for i in icd_in_data if i not in list( icd_description["ICD9_CODE"] ) ]
print ( len ( z ) )

print ('num in data but not in full preferred label, should be zero ?? ')
z = [ i for i in icd_in_data if i not in Preferred_Label ]
print (len(z))

print ('num in description file but not in data, should be zero ?? ')
w = [ i for i in list(icd_description["ICD9_CODE"]) if i not in icd_in_data ] 
print ( len ( w ) ) 

## okay to see something not overlap because the labeling in the mimic data has some strange cases. 
# desc_from_mimic_db = set ( list ( Preferred_Label.keys() ) ) 
# icd_in_data = set ( icd_in_data ) 
# icd_in_data_with_desc = list( icd_in_data.intersection( desc_from_mimic_db ) )
# icd_in_data_with_desc.sort() 


print ('\nnum of labels found in the mimic3 (not all complete labels) {}\n'.format(len(icd_in_data)) ) 

label_index_map = {} 
reverse_label_index_map = {}

description = {}
description_in_indexing = {} 
description_in_indexing_arr = []

label_name_arr = [] 

print ('go through the icd mimic file, and then keep icd labels that have descriptions, we will only use these')

counter = 0 ## @counter is needed so that we can do proper indexing for the labels in the data, and also have the description 

icd_description = icd_description.loc[icd_description['ICD9_CODE'].isin(icd_in_data)]
icd_ = list ( icd_description ['ICD9_CODE'] ) 


# SPECIAL_MAP_ICD = {'11.8':'011.8', '23.9':'023.9', '719.70':'719.7', '17.0':'17'}
SPECIAL_MAP_ICD = {'719.70':'719.7'}

for i in range(len(icd_in_data)) : 

  ## !! important: 
  ## use the @icd_in_data_with_desc to track icd9 with valid label 
  ## this icd9 not used in the mimic data. so we skip it. we will use the @Preferred_Label label
  this_icd = icd_in_data[i]
  
  if this_icd in SPECIAL_MAP_ICD: 
    this_icd = SPECIAL_MAP_ICD[this_icd] ## map strange naming convention 

  if this_icd in label_index_map: ## because we will see label '719.7' in @SPECIAL_MAP_ICD twice ?? 
    print ('duplicated label {}'.format(this_icd))
    continue

  if (this_icd not in Preferred_Label) and (this_icd not in icd_description):
    print ('no english desc ??')
    print (this_icd)
    continue

  if this_icd not in icd_ : 
    description [ this_icd ] = Preferred_Label[ this_icd ] 
  else : 
    description [ this_icd ] = icd_description[ icd_description['ICD9_CODE']==this_icd]['LONG_TITLE'].tolist()[0]
  
  desc = Punctuation (description [ this_icd ]) 
  desc = desc.split() 
  desc = [ d.strip() for d in desc if d not in stop_word ]
  for d in desc: ## expand vocab list 
    if d not in vocab_index_map: 
      vocab_index_map[d] = len(vocab_index_map)
      reverse_map[len(vocab_index_map)] = d
  

  indexing = [vocab_index_map[d] for d in desc if d in vocab_index_map ]
  description_in_indexing [ this_icd ] = indexing
  description_in_indexing_arr.append( indexing ) # @description_in_indexing_arr is used to convert into tensor 
  label_name_arr.append( this_icd ) 

  # keep the label 
  label_index_map[ this_icd ] = counter
  reverse_label_index_map[ counter ] = this_icd
  counter = counter + 1 


print ('\ntotal num of icd with description in both bioportal and database {} , we will use this as num label for prediction ? \n'.format(counter))

## append the rest at the end so that label that appears show up correctly. 

xx = list(Preferred_Label.keys()) ## merge to the whole tree
xx.sort()
for p in xx: 
  if p not in label_index_map: 

    description [ p ] = Preferred_Label[p] ## english description 

    desc = Punctuation ( description [ p ] ) 
    desc = desc.split() 
    desc = [ d.strip() for d in desc if d not in stop_word ]

    for d in desc: ## expand vocab list 
      if d not in vocab_index_map: 
        vocab_index_map[d] = len(vocab_index_map)
        reverse_map[len(vocab_index_map)] = d

    indexing = [vocab_index_map[d] for d in desc if d in vocab_index_map ]

    description_in_indexing[p] = indexing
    description_in_indexing_arr.append ( indexing )
    label_name_arr.append(p)
    new_len = len(label_index_map)
    label_index_map[p] = new_len
    reverse_label_index_map[new_len] = p
    

print ('\ntotal num of icd after appending to bioportal {}\n'.format(len(label_index_map)))


fout = open ('vocab+icd_index_map.txt','w')
key = list ( reverse_map.keys() )
key.sort() 
print (reverse_map[0] ) ## see padding and unknown 
print (reverse_map[1] ) ## see padding and unknown 

for k in key: ## 
  fout.write( reverse_map[k] + "\n") # retain the ordering 
fout.close() 

pickle.dump( description_in_indexing , open("description_in_indexing.pickle","wb") )
pickle.dump( description , open("description.pickle","wb") )
pickle.dump( description_in_indexing_arr , open("description_in_indexing_arr.pickle","wb") )

pickle.dump( vocab_index_map , open("vocab+icd_index_map.pickle","wb") )
pickle.dump( reverse_map , open("vocab+icd_reverse_map.pickle","wb") )

pickle.dump( label_index_map , open("label_index_map.pickle","wb") )
pickle.dump( reverse_label_index_map , open("reverse_label_index_map.pickle","wb") )

pickle.dump( label_name_arr , open("label_name_arr.pickle","wb") )


##

print ('\n\ncreate emb. for labels desc. using 300 dim. \n\n')
word_dim = 300
pretrain_emb = np.zeros ( (len(vocab_index_map), word_dim ) ) 

fin = open ("/u/flashscratch/d/datduong/w2vModel1Gram9Jan2019/w2vModel1Gram9Jan2019.txt",'r')
counter = 0 
for line in fin: 
  if counter == 0 : 
    counter = 1 # skip line 1
    continue
  line = line.strip().split() 
  word = line[0]
  if word in vocab_index_map:
    pretrain_emb [ vocab_index_map[word] ] = np.array ( line[1:len(line)] )


pickle.dump(pretrain_emb , open("pubmed_go_data_vocab_indexing_prune_pretrain_emb_300.pickle","wb")) ## later used to load into the emb

print ('num of words and dim ')
print (pretrain_emb.shape)

print ('sample')
print (pretrain_emb[0:5])



print ('\n\ncreate emb. for labels desc. using 100 dim. \n\n')
word_dim = 100
pretrain_emb = np.zeros ( (len(vocab_index_map), word_dim ) ) 

fin = open ("/u/flashscratch/d/datduong/w2vModel1Gram11Nov2018/w2vModel1Gram11Nov2018.txt",'r')
counter = 0 
for line in fin: 
  if counter == 0 : 
    counter = 1 # skip line 1
    continue
  line = line.strip().split() 
  word = line[0]
  if word in vocab_index_map:
    pretrain_emb [ vocab_index_map[word] ] = np.array ( line[1:len(line)] )


pickle.dump(pretrain_emb , open("pubmed_go_data_vocab_indexing_prune_pretrain_emb_100.pickle","wb")) ## later used to load into the emb

print ('num of words and dim ')
print (pretrain_emb.shape)

print ('sample')
print (pretrain_emb[0:5])

