import csv
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

print("Data Loading")
df  = pd.read_csv('./data/ATCPTLabeled_StringSplit.csv', names=['sex','1','2','3','4','5','6','7','ae','a','b','c','d','label'], header=None)#, nrows=1000)

print(" (1/4)Data Loaded")

cols = df.columns.to_list()
cols = cols[1:] + cols[:1]
df = df[cols]



#print(df)
print(" (2/4)Changed position of sex.")

le=LabelEncoder()
for col in df[['1','4','5']]:
    df[col]=le.fit_transform(df[col])
print(" (3/4)Label Encoded.")

#df['ae']=df['ae'].apply(lambda x:str(x).zfill(7))


csvPath = './data/ATCPTLabeled_LabelEncoder_ForATC.csv'
df.to_csv(csvPath,header=False,index=None)  #save csv file with dataframe object 
print("(4/4)Completed.")
