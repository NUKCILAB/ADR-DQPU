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
data = pd.read_csv('.\data\ATCPTLabeled_OHE_Train_Balance_4000000.csv', header=None)
print("DL")
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
data = shuffle(data,random_state = 42)
print("shuffled")
csvPath = './data/ATCPTLabeled_OHE_Train_Balance_4000000_shuffled.csv'
data.to_csv(csvPath,header=False,index=None,mode='a')
print("Done")
