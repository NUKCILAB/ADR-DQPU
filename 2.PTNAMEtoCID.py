import csv
import os

temp0=[]
temp1=[]
temp2=[]
temp3=[]
temp5=[]
temp6=[]
temp7=[]
temp8=[]
temp9=[]

with open('./data/ATCPT.csv') as f1:
    
    f1_csv = csv.reader(f1)
    for row in f1_csv:
        temp0.append(row[0])       
        temp5.append(row[1])
        temp6.append(row[3])        
        temp7.append(row[4])        
        temp8.append(row[5])
        temp9.append(row[6])
        temp = row[2].replace(' ','')
        temp=temp.lower()
        temp1.append(temp)
print("File 1 is loaded.")
with open('./data/meddra_PT.csv') as f2:  #讀實驗室的pt id
        
    f2_csv = csv.reader(f2)
    for row in f2_csv:
        temp2.append(row[0])
        temp = row[3].replace(' ','')
        temp=temp.lower()
        temp3.append(temp)
print("File 2 is loaded.")
with open('./data/ATCPTCID.csv', 'w', newline='') as csvfile: #將兩表link，得到ATC 對應PT_ID的組合表

    writer = csv.writer(csvfile)
    
    for i in range(0,len(temp0)):
        flag = 0
        for j in range(0,len(temp3)):
            if temp1[i] == temp3[j]:
                writer.writerow([temp0[i], temp5[i], temp2[j], temp6[i], temp7[i], temp8[i], temp9[i]])
                flag =1
                break

print("Ccompleted.")
