
import pickle, gzip, os, sys, re
import random 
from random import shuffle
import numpy as np 
import pandas as pd

## notice some error in label formatting. 

# has no 996, but the formated string data says 996 is in it. 

# 601478,89148,145375,1,"44101"
# 601479,89148,145375,2,"5185"
# 601480,89148,145375,3,"431"
# 601481,89148,145375,4,"34982"
# 601482,89148,145375,5,"4275"
# 601483,89148,145375,6,"99812"
# 601484,89148,145375,7,"5849"
# 601485,89148,145375,8,"2930"
# 601486,89148,145375,9,"2851"
# 601487,89148,145375,10,"29630"
# 601488,89148,145375,11,"45382"
# 601489,89148,145375,12,"9972"
# 601490,89148,145375,13,"99702"
# 601491,89148,145375,14,"42732"
# 601492,89148,145375,15,"9971"
# 601493,89148,145375,16,"99731"
# 601494,89148,145375,17,"9980"
# 601495,89148,145375,18,"2760"
# 601496,89148,145375,19,"9982"
# 601497,89148,145375,20,"4019"
# 601498,89148,145375,21,"30393"
# 601499,89148,145375,22,"78060"
# 601500,89148,145375,23,"42731"
# 601501,89148,145375,24,"30000"
# 601502,89148,145375,25,"28860"
# 601503,89148,145375,26,"2875"
# 601504,89148,145375,27,"53081"
# 601505,89148,145375,28,"V1046"
# 601506,89148,145375,29,"V4572"
# 601507,89148,145375,30,"V4986"
# 601508,89148,145375,31,"V1005"
# 601509,89148,145375,32,"E8782"
# 601510,89148,145375,33,"E9342"
# 601511,89148,145375,34,"V4987"

# 201967,89148,145375,1,"3845"
# 201968,89148,145375,2,"311"
# 201969,89148,145375,3,"3479"
# 201970,89148,145375,4,"9672"
# 201971,89148,145375,5,"3749"
# 201972,89148,145375,6,"3522"
# 201973,89148,145375,7,"362"
# 201974,89148,145375,8,"7761"
# 201975,89148,145375,9,"3791"
# 201976,89148,145375,10,"9672"
# 201977,89148,145375,11,"9960" ## why repeat ?? 
# 201978,89148,145375,12,"4311"
# 201979,89148,145375,13,"966"
# 201980,89148,145375,14,"9605"
# 201981,89148,145375,15,"3961"
# 201982,89148,145375,16,"3893"
# 201983,89148,145375,17,"8872"
# 201984,89148,145375,18,"9961" ## why repeat ?? actually, they are not parents/children 
# 201985,89148,145375,19,"9604"


## also notice, this is not true ?? because some code may have error in formatting ? 
## >>> len(df['ICD9_CODE'].unique()) 
## 8994

os.chdir ('/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019')

each_person_icd = pickle.load(open("/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_PROCEDURES_ICD_dot_format_to_leaf.pickle","rb"))
Preferred_Label = pickle.load(open("Preferred_Label.pickle","rb"))
Label_Desc = pickle.load(open("description_in_indexing.pickle","rb"))

list_data = ['disch_dev_split',
             'disch_full',
             'disch_test_split',
             'disch_train_split']

for data_type in list_data : 

  data_in = pd.read_csv( "../"+data_type+".csv", dtype=str )
  # data_in = data_in.dropna()

  correct_label = [ ]

  row_iterator = data_in.iterrows() ## is it faster ?? 
  for i, row in row_iterator:
    # label = list(set( row['LABELS'].strip().split(";") ))  
    person = row['SUBJECT_ID']+ " " + row['HADM_ID']
    if person not in each_person_icd: 
      correct_label.append ( 'null' ) ## has no icd9 ?? 
    else: 
      label = each_person_icd[person] ## replace the label with the label-file 
      label = [ l for l in label if (l in Preferred_Label) or (l in Label_Desc) ] ## must have description 
      correct_label.append ( ";".join ( str(l) for l in label ) )

  # 
  data_in['LABELS'] = correct_label
  data_in.to_csv( data_type+"_correct_icd.csv",index=False ) # "disch_full_correct_icd.csv"







