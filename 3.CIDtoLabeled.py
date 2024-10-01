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
temp4=[]

with open('./data/ATCPTCID_withoutOther.csv') as f1:
    
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
with open('./data/sider_pt_ATC.csv') as f2:  
        
    f2_csv = csv.reader(f2)
    for row in f2_csv:
        temp2.append(row[0])
        temp = row[2].replace(' ','')
        temp=temp.lower()
        temp3.append(temp)
        temp4.append(row[1])
with open('./data/ATCPTLabeled.csv', 'w', newline='') as csvfile: 

    writer = csv.writer(csvfile)
    checkpoint = 1
    for i in range(0,len(temp0)):
        flag = 0
        if i == checkpoint :
            print(checkpoint)
            checkpoint=checkpoint+100000
        for j in range(0,len(temp3)):
            if temp5[i] == temp2[j]:
              if temp1[i] == temp3[j]:
                writer.writerow([temp0[i], temp5[i], temp4[j], temp6[i], temp7[i], temp8[i], temp9[i],1])
                flag =1
                break
            if j == len(temp3)-1 and flag == 0:
                writer.writerow([temp0[i], temp5[i], temp4[j], temp6[i], temp7[i], temp8[i], temp9[i],0])


print("Completed.")
