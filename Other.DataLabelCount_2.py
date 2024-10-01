import csv
import os
import gc
import time
import pandas as pd
import numpy as np
from collections import Counter

reader   = pd.read_csv('./data/ATCPTLabeled_OneHotEncoder_ForATC.csv', header=None, iterator=True)
print("Iteration loaded")
#------------------------------------------------------------------------------------------
num =  10000
loop = True
data = pd.DataFrame()
P=0
U=0
N=0
count_round=0
start = time.process_time()
while loop:
    try:
        df = reader.get_chunk(num)
        count_A = Counter(df.iloc[: , -1])
        P=P+count_A[1]
        U=U+count_A[0]
        count_round+=1
        print("Round : ",count_round)

    except StopIteration:
        end = time.process_time()
        print("RunTime : %f s" % (end - start))
        print("Done ! ", "P=",P,"U=", U ,"All=" , (P+U))
        print("P=",round(P/(P+U),4),round(U/(P+U),4),"=U")
        loop = False
        
#------------------------------------------------------------------------------------------
print("All Done.")

