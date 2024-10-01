import csv
import os
import gc
import time
import math
import random
import winsound
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
np.seterr(divide='ignore', invalid='ignore')
random.seed(2)
np.random.seed(2)
#------------------------------------------------------------------------------------------
emax=1
emin=0.01
ed=0.999953
count=0
while(True):
    count+=1
    emax=emax*ed
    if emax<emin:
        break
#200000
print(count)
