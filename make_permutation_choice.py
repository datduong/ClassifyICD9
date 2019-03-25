
import os,sys,pickle
import numpy as np
from pyDOE import *

permutation_choice = {}
num_label = [2,3,4,5,6,10]
for n in num_label : 
  permutation_choice[n] = fullfact(np.repeat(2,n))


pickle.dump ( permutation_choice, open("permutation_choice.pickle","wb"))

