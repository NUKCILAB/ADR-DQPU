import csv
import os

temp0=[]
temp1=[]
temp2=[]
temp3=[]
temp4=[]
temp5=[]
temp6=[]
temp7=[]
temp8=[]
temp9=[]



print("Data Loading")
with open('.\data\ATCPTLabeled.csv') as f1:
    
    f1_csv = csv.reader(f1)
    for row in f1_csv:
        temp0.append(row[0])   
        temp1.append(row[1])   
        temp2.append(row[2])   
        temp3.append(row[3])   
        temp4.append(row[4])   
        temp5.append(row[5])   
        temp6.append(row[6])   
        temp7.append(row[7])    
print("Data Loaded")

print("GOGOGO")

with open('.\data\ATCPTLabeled_StringSplit.csv', 'w', newline='') as csvfile: 

    writer = csv.writer(csvfile)
    checkpoint = 0
    for i in range(0,len(temp0)):
        #print([temp0[i], temp1[i][0], temp1[i][1]+temp1[i][2],temp1[i][3],temp1[i][4], temp1[i][5]+temp1[i][6],temp2[i][1:], temp3[i], temp4[i], temp5[i], temp6[i],temp7[i]])
        writer.writerow([temp0[i], temp1[i][0], temp1[i][1],temp1[i][2],temp1[i][3],temp1[i][4], temp1[i][5],temp1[i][6],temp2[i][1:], temp3[i], temp4[i], temp5[i], temp6[i],temp7[i]])
        if i == checkpoint :
            print(checkpoint)
            checkpoint=checkpoint+100000
print("Completed.")
