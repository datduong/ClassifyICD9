
import pickle, gzip, os, sys, re
import random 
from random import shuffle
import numpy as np 
import pandas as pd

sys.path.append ( '/u/flashscratch/d/datduong/ICD9multitask/' )
import helper

os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')


## parents/children look up index 
label_index_map = pickle.load( open("label_index_map.pickle","rb") )
reverse_label_index_map = pickle.load( open("reverse_label_index_map.pickle","rb") )

parents = pickle.load ( open("parent_icd.pickle","rb") ) 
children = pickle.load ( open("children_icd.pickle","rb") ) 


## it is only sensible to make prediction based on leaf nodes (most specific)
leaf_node = [] 
leaf_node_index = []
for j in label_index_map: 
  if (j in parents) and (j not in children) : ## leaf will not have children 
    leaf_node.append ( j )
    leaf_node_index.append ( label_index_map[j] )


pickle.dump ( leaf_node_index, open("leaf_node_index.pickle","wb"))
print (len(leaf_node))

## do all the people have leaf node ?? 
not_in_leaf_node = {}
each_person_label = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_PROCEDURES_ICD_dot_format_to_leaf.pickle","rb"))
for k , val in each_person_label.items() :  
  x = [i for i in val if i not in leaf_node]
  if len (x)>0: 
    not_in_leaf_node[k] = x 


print ('people found in mimic data with nodes that are not leaf {}'.format(len(not_in_leaf_node) ) )
pickle.dump( not_in_leaf_node, open("not_in_leaf_node.pickle",'wb') ) 

## what do we do with these ?? 

# leaf_node_index = pickle.load (open("leaf_node_index.pickle","rb") ) 
# not_in_leaf_node = pickle.load( open("not_in_leaf_node.pickle",'rb') ) 

# code_not_leaf = []
# for k, val in not_in_leaf_node.items() : 
#   for v in val: 
#     if v not in code_not_leaf: 
#       code_not_leaf.append(v)

# print ('\nnum of icd that are not leaf nodes')
# print ( len(code_not_leaf) ) 


## ...

# convert_to_leaf ('519.1',children,Preferred_Label)
# '519.19'
# convert_to_leaf ('779.3',children,Preferred_Label)
# '779.31'

# Preferred_Label = pickle.load ( open("Preferred_Label.pickle","rb") ) 

# Preferred_Label['519.1']

# Preferred_Label [ '453.8' ]
# for c in children['453.8']:
#   print (Preferred_Label[c])

# '453.8'

# ['453.8', '596.8', '518.5']

# 5353,"45381","Ac embl suprfcl up ext","Acute venous embolism and thrombosis of superficial veins of upper extremity"
# 5354,"45382","Ac DVT/embl up ext","Acute venous embolism and thrombosis of deep veins of upper extremity"
# 5355,"45383","Ac emblsm up ext NOS","Acute venous embolism and thrombosis of upper extremity, unspecified"
# 5356,"45384","Ac emblsm axillary veins","Acute venous embolism and thrombosis of axillary veins"
# 5357,"45385","Ac embl subclav veins","Acute venous embolism and thrombosis of subclavian veins"
# 5358,"45386","Ac embl internl jug vein","Acute venous embolism and thrombosis of internal jugular veins"
# 5359,"45387","Ac embl thorac vein NEC","Acute venous embolism and thrombosis of other thoracic veins"
# 5360,"45389","Ac embolism veins NEC","Acute venous embolism and thrombosis of other specified veins"


# print (len(not_in_leaf_node))

# not_in_leaf_node 

# for k in ['453.8', '596.8', '518.5']: 
#   convert_to_leaf (k,children)

