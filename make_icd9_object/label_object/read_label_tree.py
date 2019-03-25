import pandas as pd 
import re, pickle, sys, os

sys.path.append('/u/flashscratch/d/datduong/ICD9multitask')
# import helper 


os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

## !! REMOVE ROOT ??? 001-999.99 AND 00-99.99 ??? MAY BE NOT .... 

parent_icd = {}
Preferred_Label = {}

icd_in = pd.read_csv("ICD9CM.csv", dtype=str)

for i in range(icd_in.shape[0]):  
  this_node = icd_in.loc[i,'Class ID'].strip().split("/")[-1]
  Preferred_Label[this_node] = icd_in.loc[i,'Preferred Label'].lower()
  try: 
    parent = icd_in.loc[i,'Parents'].strip().split("/")[-1] ## last element ... example: http://purl.bioontology.org/ontology/ICD9CM/784.4
    if parent=='owl#Thing': ## why the hell are these things here ?? 
      continue
    # if "-" not in parent: ## may see this '520-529.99'
    #   parent = helper.reformat_icd_str(parent) ## put back the standard dot
    if this_node not in parent_icd: 
      parent_icd[this_node] = [ parent ]
    else: 
      parent_icd[this_node].append( parent )
  except: 
    pass 
  # if this_node == '520-529.99':
  #   print (icd_in.iloc[i]) ## notice parents are group http://purl.bioontology.org/ontology/ICD9CM/520-529.99
  #   break 

# 

key_out = list ( Preferred_Label.keys() ) 
key_out.sort() 
fout = open( 'Preferred_Label.txt','w') 
for k in key_out: 
  fout.write(k + "\t" + Preferred_Label[k]  + '\n' )

fout.close()  

pickle.dump ( Preferred_Label, open("Preferred_Label.pickle","wb") ) 
pickle.dump ( parent_icd, open("parent_icd.pickle","wb") ) 

## make children edges 
children_icd = {}
pp = list ( parent_icd.keys() ) 
pp.sort() 

for p in pp: ## for each node that has a parent 
  parent_of_p = parent_icd[p] ## the parents of node p
  for k in parent_of_p: ## for each of the parent found, we add p as its child 
    if k=='owl#Thing': 
      continue
    if k not in children_icd: 
      children_icd[k] = [p]
    else: 
      children_icd[k].append(p)

#

pickle.dump ( children_icd, open("children_icd.pickle","wb") ) 


## get all ancestors 

from copy import deepcopy

parent_icd = pickle.load ( open("parent_icd.pickle","rb") ) 
Preferred_Label = pickle.load ( open("Preferred_Label.pickle","rb") ) 

all_ancestors = {}
for node in Preferred_Label: ## get all ancestors 
  #
  if node not in parent_icd: ## has no parent ??
    continue
  #
  all_ancestors[node] = deepcopy(parent_icd[node]) ## the parents 
  list_iter_through = deepcopy(parent_icd[node])
  final_list = deepcopy(parent_icd[node])
  while len(list_iter_through) > 0: 
    for p in list_iter_through: 
      list_iter_through.remove(p) ## remove p
      if p in parent_icd: 
        final_list = final_list + parent_icd [p] ## p has parent, then add the parent 
        list_iter_through = list_iter_through + parent_icd [p] ## add parents to @list_iter_through
  #
  # recore all ancestors 
  all_ancestors[node] = final_list 
  

pickle.dump ( all_ancestors, open("all_ancestors_icd.pickle","wb") ) 

 

