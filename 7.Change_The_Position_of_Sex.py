import csv
import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


reader   = pd.read_csv('./data/ATCPTLabeled_OneHotEncoder_ForATC_Temp.csv', header=None, iterator=True)
print("Iteration loaded")

flag = 0
loop = True

while loop:
    try:
        
        df = reader.get_chunk(10000000)
        #print(df)
        cols = df.columns.to_list()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        #print(df)
        csvPath = './data/ATCPTLabeled_OneHotEncoder_ForATC.csv'
        df.to_csv(csvPath,header=False,index=None,mode='a')  #save csv file with dataframe object
        flag=flag+1
        print("Round " , flag, " has been completed." )
            
    except StopIteration:
        print ("Iteration is stopped.")
        del reader
        gc.collect()
        loop= False


print("All done ! ")

