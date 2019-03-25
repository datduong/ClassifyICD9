

## for testing, select a few labels 
import pickle, gzip, os, sys, re
import random 
from random import shuffle
import numpy as np 
import pandas as pd

def submitJobs ( top_k, add_full_tree, do_manual, do_small ) : ## @add_full_tree add all ancestors 

  os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

  all_ancestors = pickle.load ( open("all_ancestors_icd.pickle","rb") )

  list_data = ['disch_dev_split_correct_icd',
              #  'disch_full_correct_icd',
              'disch_test_split_correct_icd',
              'disch_train_split_correct_icd']


  print ('\n\ntesting on subset ??\n')
  if do_manual == 0: 
    to_use = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_top"+str(top_k)+".pickle","rb"))

  if do_manual == 1: 
    # to_use = ['401.9','38.93','428.0','427.31','414.01']
    # to_use = ['38.93','584.9'] ## add these KNOWN anti-correlated ones '584.5','584.6','584.7','584.8'
    to_use = ['38.93','584.5','584.6','584.7','584.8','584.9'] 
    top_k = len(to_use)
    if do_small == 'do_small': 
      top_k = "small_"+str(top_k) 

    pickle.dump (to_use, open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_manual_"+str(top_k)+".pickle","wb"))
  
  to_use = list (set (to_use))
  to_use.sort() 
  print ( 'label wanted ')
  print ( to_use )


  if add_full_tree == 1: 
    ancestor = []
    for j in to_use:
      ancestor = ancestor + all_ancestors[j]
    
    ancestor = list (set (ancestor))
    ancestor.sort() 
    to_use_with_anc = to_use + ancestor ## add ancestors ?? 
    to_use_with_anc = list (set (to_use_with_anc))
    ## these ICD+ancestors of the ICD will be later used to extract from the whole 25k list. 

    to_use_with_anc.sort() 
    
    pickle.dump (to_use_with_anc, open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_top_full_tree_"+str(top_k)+".pickle","wb"))
    print ( '\nlabel wanted + their ancestors')
    print ( to_use_with_anc ) 


  for text_file in list_data: 

    df = pd.read_csv (text_file+".csv",dtype=str) # 'disch_test_split.csv'
    df['LABELS'] = df['LABELS'].fillna('null')

    row_iterator = df.iterrows() ## is it faster ?? 
    keep_row = []
    for i, row in row_iterator:
      label = set(row['LABELS'].split(";")) ## get unique, because there are duplicated ... why ??
      inter = label.intersection ( to_use ) ## select only the LEAF NODES THAT WE WANT. ... LATER WE WILL APPEND THE PARENTS TO THEM 
      
      if len(inter) > 0 : 

        if do_small == 'do_small': 
          if np.random.uniform() < .7 : 
            continue ## skip so we get smaller dataset 

        full_icd = list(inter)
        full_icd.sort() 

        if add_full_tree == 1: 
          for j in inter: ## for each label, we get their whole ancestors 
            full_icd = full_icd + all_ancestors[j] 
          
          ## add ancestors (may have duplicated so take "set")
          full_icd = list (set ( full_icd ) )
          full_icd.sort()

        df.loc[i,'LABELS'] = ";".join(j for j in full_icd) ## replace 
        keep_row.append(i)

    # keep these row only 
    df = df.loc[keep_row]
    
    if add_full_tree == 1: 
      df.to_csv( text_file+"_full_tree_"+str(top_k)+"icd.csv", index=False )
    else: 
      if do_manual == 1: 
        df.to_csv( text_file+"_manual_"+str(top_k)+"icd.csv", index=False )
      else: 
       df.to_csv( text_file+"_"+str(top_k)+"icd.csv", index=False )


      
if len(sys.argv)<1: ## run script 
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( int(sys.argv[1]) , int(sys.argv[2]) , int(sys.argv[3]), sys.argv[4] ) 
	

