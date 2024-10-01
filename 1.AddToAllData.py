import csv
import os
from tensorflow.keras.optimizers import SGD
from datetime import date
from sklearn.linear_model import LogisticRegression
from shutil import copyfile


year = 2004
flagflag = 0
while year <= 2019:
    mm = 1
    if flagflag == 0:
      mm =1
      flagflag = 1
    while mm <= 12:
      if year == 2019 and mm == 4:
        break
      mmtwo=('{0:02d}'.format(mm))
      print(year,mmtwo)
      #還原PT_Key To PT_Name
      import csv
      temp0=[]
      temp1=[]
      temp2=[]
      temp3=[]
      temp5=[]
      temp6=[]
      temp7=[]
      temp8=[]
      temp9=[]

      with open('/content/drive/MyDrive/SignalDetectionOfADRs/Data/dbo.AGE_L3,ATC_L5,PT,'+str(year)+mmtwo+'.Table.csv') as f1:
          
          f1_csv = csv.reader(f1)
          for row in f1_csv:
              temp0.append(row[0]) 
              temp = row[1].replace(' ','')      
              temp5.append(temp)
              temp6.append(row[3])        
              temp7.append(row[4])        
              temp8.append(row[5])
              temp9.append(row[6])
              temp = row[2].replace(' ','')
              temp=temp.lower()
              temp1.append(temp)
      with open('/content/drive/MyDrive/SignalDetectionOfADRs/Data/PT_Key_To_Name_Server.csv') as f2:  #讀實驗室的pt id
              
          f2_csv = csv.reader(f2)
          for row in f2_csv:
              temp2.append(row[0])
              temp3.append(row[1])
      with open('/content/drive/MyDrive/SignalDetectionOfADRs/Data/ATCPT.csv', 'a+', newline='') as csvfile: #將兩表link，得到ATC 對應PT_ID的組合表

          writer = csv.writer(csvfile)
          
          for i in range(0,len(temp0)):
              flag = 0
              for j in range(0,len(temp3)):
                  if temp1[i] == temp3[j]:
                      writer.writerow([temp0[i], temp5[i], temp2[j], temp6[i], temp7[i], temp8[i], temp9[i]])
                      flag =1
                      break
      #            if j == len(temp3)-1 and flag == 0:
      #                writer.writerow([temp0[i], temp5[i], 0, temp6[i], temp7[i], temp8[i], temp9[i]])

      print(year,mmtwo,"Completed.")
      mm=mm+1
    year=year+1
     
