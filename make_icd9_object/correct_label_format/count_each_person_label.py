

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle 
import pandas as pd 


each_person_label = {}

person_full_icd = pd.read_csv( '/u/flashscratch/d/datduong/MIMIC3database/disch_full_correct_icd.csv', dtype=str )
row_iterator = person_full_icd.iterrows()
for i,row in row_iterator: # "SUBJECT_ID","HADM_ID"
  this_rec = row['SUBJECT_ID'] + " " + row['HADM_ID']
  if this_rec not in each_person_label: 
    each_person_label[this_rec] = [row['LABELS']]
  else: 
    each_person_label[this_rec].append ( row['LABELS'] ) 


pickle.dump ( each_person_label, open("/u/flashscratch/d/datduong/MIMIC3database/disch_full_each_person_label_correct_icd.pickle","wb"))


## count frequency of labels .

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle 
import pandas as pd 
import heapq

each_person_label = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/disch_full_each_person_label_correct_icd.pickle","rb"))

label_count = {} 
for p in each_person_label: 
  for l in each_person_label[p]: 
    if l not in label_count: 
      label_count[l] = 1
    else: 
      label_count[l] = label_count[l] + 1 

pickle.dump ( label_count, open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_count.pickle","wb"))

## take top k 

label_count = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_count.pickle","rb"))

k = 2000 ## see some top @k 
icd9_count_sorted = heapq.nlargest(k, label_count, key=label_count.get)
name = []
for i in icd9_count_sorted: 
  print ('name {} , value {}'.format(i , label_count[i] ) ) 
  name.append(i)


pickle.dump ( name, open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/label_top"+str(k)+".pickle","wb"))
