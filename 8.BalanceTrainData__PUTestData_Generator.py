import csv
import os
import gc
import time
import winsound
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


#reader   = pd.read_csv('./data/ATCPTLabeled_OneHotEncoder_ForATC.csv', header=None, iterator=True)
#reader   = pd.read_csv('./data/ATCPTLabeled_OHE_Train_Balance_4000000_shuffled.csv', header=None, iterator=True)
reader   = pd.read_csv('./data/TRAININGQ.csv', header=None, iterator=True)
print("Iteration loaded")
#------------------------------------------------------------------------------------------
steps = 10000
#For Train Generator
need =  200000#here
half_need = need/2
#For Test Generator
test_need = 0
half_test_need = test_need/2

#------------------------------------------------------------------------------------------
#Train DATA Generator
try:
    os.remove('./data/ATCPTLabeled_OHE_Train_Balance_Q'+str(need)+'.csv')
except OSError:
    print("")

check = -1
point = True
Pflag=0
Uflag=0
start = time.process_time()

while point:
    df = reader.get_chunk(steps)
    check+=1
    data = pd.DataFrame()
    for i in range(0,len(df)):
        if df.iloc[: , -1][i+(steps*check)] == 0:
            if Uflag < half_need :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Uflag+=1
        else:
            if Pflag < half_need :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Pflag+=1

    csvPath = './data/ATCPTLabeled_OHE_Train_Balance_'+str(need)+'.csv'
    data.to_csv(csvPath,header=False,index=None,mode='a')

    if Pflag+Uflag == need:
        end = time.process_time()
        print("RunTime : %f s" % (end - start))
        print("New data count : " , (Pflag+Uflag),"/",need)
        print("Train Data Generator Completed.")
        point = False
    else:
        print("Train Data not enough ! ",  "P=",Pflag,"U=", Uflag, "All=",(Pflag+Uflag),"/",need)
    
print(" ")

#------------------------------------------------------------------------------------------
''''#TEST DATA Generator

try:
    os.remove('./data/ATCPTLabeled_OHE_Test_PU_'+str(test_need)+'.csv')
except OSError:
    print("")
    

point = True
Pflag=0
Uflag=0
start = time.process_time()

while point:
    df = reader.get_chunk(steps)
    check+=1
    data = pd.DataFrame()
    for i in range(0,len(df)):
        if df.iloc[: , -1][i+(steps*check)] == 0:
            if Uflag <4452600 :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Uflag+=1
        else:
            if Pflag < 338623 :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Pflag+=1

    csvPath = './data/ATCPTLabeled_OHE_Test_PU_'+str(test_need)+'.csv'
    data.to_csv(csvPath,header=False,index=None,mode='a')


    if Pflag+Uflag == test_need:
        end = time.process_time()
        print("RunTime : %f s" % (end - start))
        print("New data count : " , (Pflag+Uflag),"/",test_need)
        print("Test Data Generator Completed.")
        point = False
    else:
        print("Test Data not enough ! ",  "P=",Pflag,"U=", Uflag, "All=",(Pflag+Uflag),"/",test_need)
'''
print(" ")
winsound.Beep(2000,1000)
print("All Done.")

