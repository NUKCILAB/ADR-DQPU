import csv
import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer


reader   = pd.read_csv('./data/ATCPTLabeled_LabelEncoder_ForATC.csv', names=['1','2','3','4','5','6','7','ae','a','b','c','d','label','sex'], header=None, iterator=True)
print("Iteration loaded")

try:
    os.remove('./data/ATCPTLabeled_OneHotEncoder_ForATC_Temp.csv')
except OSError:
    print("")

flag = 0
loop = True

while loop:
    try:
        #print("Chunk Data Loading")
        df = reader.get_chunk(10000000)
        npdf = df.to_numpy()
        newdf = pd.DataFrame(npdf, columns=['1','2','3','4','5','6','7','ae','a','b','c','d','label','sex'])
        del npdf
        #print("Chunk Data Loaded")
        enc = OneHotEncoder(sparse_output=False)
        columns_to_one_hot = ['1','2','3','4','5','6','7']
        encoded_array = enc.fit_transform(df.loc[:,columns_to_one_hot])
        df_encoded = pd.DataFrame(encoded_array,columns=enc.get_feature_names_out() )
        df = pd.concat([df_encoded,newdf],axis=1)
        df.drop(labels= columns_to_one_hot,axis=1,inplace=True)
        print(df)
        #print("Chunk One Hot Encoded")
        csvPath = './data/ATCPTLabeled_OneHotEncoder_ForATC_Temp.csv'
        df.to_csv(csvPath,header=False,index=None,mode='a')  #save csv file with dataframe object 
        #print("Chunk Completed.")
        del encoded_array
        del df_encoded
        del df
        gc.collect()
        flag=flag+1
        print("Round " , flag, " has been completed." )
        print("---------------------------------------------------------------------")
    except StopIteration:
        print ("Iteration is stopped.")
        del reader
        gc.collect()
        loop= False

print("All done ! ")

