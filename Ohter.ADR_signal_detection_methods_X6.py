import csv
import os
import gc
import time
import winsound
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer

reader   = pd.read_csv('./data/TEST2019Q1_Clean.csv', header=None, iterator=True)
print("The test data loading in ram.")
#------------------------------------------------------------------------------------------
steps = 10000
data_num = 2806034 #how many data?
#------------------------------------------------------------------------------------------
#TEST 
check = -1
point = True
running=0
start = time.process_time()
#========================================
def ADRs(a,b,c,d):
        
        a = a
        b = b
        c = c
        d = d
        N = a + b + c + d
        #print(a,b,c,d)
        ADR_signal_ROR = 0
        ADR_signal_PRR = 0
        ADR_signal_BCPNN = 0
        ADR_signal_MHRA = 0
        ADR_signal_SPRT = 0
        ADR_signal_Q = 0
        
        #ROR
        try:
            ror = (a*d) / (b*c)
            se_ror = math.sqrt((1/a) + (1/b) + (1/c) + (1/d))
            ci_ror_increase = math.exp(math.log(ror) + (1.96 * se_ror))
            ci_ror_decrease = math.exp(math.log(ror) - (1.96 * se_ror))
                
            if ci_ror_decrease < ci_ror_increase:
                ci_ror = ci_ror_decrease
            else:
                ci_ror = ci_ror_increase
        except ZeroDivisionError as e: #處理異常：表示分母無限大
                ci_ror = 0
    
        if ci_ror > 1:
            ADR_signal_ROR=1
            #print("ROR有測出不良反應")

         #PRR
        try:
            prr = (a/(a+b)) / (c/(c+d))
            #print("PRR = " + str(prr))
            se_prr = math.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c+d)))
            #print("SE_PRR = " + str(se_prr))
            ci_prr_increase = math.exp(math.log(prr) + (1.96 * se_prr))
            ci_prr_decrease = math.exp(math.log(prr) - (1.96 * se_prr))
                
            if ci_prr_decrease < ci_prr_increase:
                ci_prr = ci_prr_decrease
            else:
                ci_prr = ci_prr_increase
            #print("CI_PRR = " + str(ci_prr))
        except ZeroDivisionError as e: #處理異常
            ci_prr = 0
        
        if ci_prr > 1:
        #if (a > 3 or a == 3) and (ci_prr > 1 or ci_prr == 1):
            ADR_signal_PRR+=1
            #print("PRR有測出不良反應")

        #BCPNN
        try:
            bcp_r = ((N+2) * (N+2)) / ((a+b+1) * (a+c+1))
        except ZeroDivisionError as e: #處理異常
            bcp_r = 0
            
        try:
            bcp_e_ic = ((a+1) * (N+2) * (N+2)) / ((N+bcp_r) * (a+b+1) * (a+c+1))
            if bcp_e_ic > 0:
                bcp_e = math.log(bcp_e_ic,2)
            else:
                bcp_e = 0
            #print("IC = " + str(ic))
        except ZeroDivisionError as e: #處理異常
            bcp_e = 0
            
        try:
            bcp_sd_a = (N-a-b+1) / ((a+b+1) * (N+3))
            bcp_sd_b = (N-a-c+1) / ((a+c+1) * (N+3))
            bcp_sd_r = (N-a+bcp_r-1) / ((a+1) * (1+N+bcp_r))
            if (bcp_sd_a + bcp_sd_b + bcp_sd_r)>=0:
                bcp_sd = math.sqrt(bcp_sd_a + bcp_sd_b + bcp_sd_r)
            else:
                bcp_sd = 0
        except ZeroDivisionError as e: #處理異常
            bcp_sd = 0
        
        bcp_increase = bcp_e + (1.96 * bcp_sd)
        bcp_decrease = bcp_e - (1.96 * bcp_sd)
            
        if bcp_decrease < bcp_increase:
            bcp = bcp_decrease
        else:
            bcp = bcp_increase
            
        if (bcp) > 0:
            ADR_signal_BCPNN = 1
            #print("BCPNN有測出不良反應")

  
        #MHRA
        try:
            prr_x = (a/(a+b)) / (c/(c+d))
            #print("PRR_X = " + str(prr_x))
        except ZeroDivisionError as e: #處理異常
            prr_x = 0
            
        try:
            x_a = pow(abs(a - ((a+b)*(a+c) / N)) - (1/2),2) / ((a+b)*(a+c) / N)
        except ZeroDivisionError:
            x_a = 0

        try:
            x_b = pow(abs(b - ((a+b)*(b+d) / N)) - (1/2),2) / ((a+b)*(b+d) / N)
        except ZeroDivisionError:
            x_b = 0
        
        try:
            x_c = pow(abs(c - ((a+c)*(c+d) / N)) - (1/2),2) / ((a+c)*(c+d) / N)
        except ZeroDivisionError:
            x_c = 0
        
        try:
            x_d = pow(abs(d - ((b+d)*(c+d) / N)) - (1/2),2) / ((b+d)*(c+d) / N)
        except ZeroDivisionError:
            x_d = 0
        
        x = x_a + x_b + x_c + x_d
        #print("PRR_X_2 = " + str(x))
        if (a > 3 or a == 3) and (prr_x > 2 or prr_x == 2) and (x > 4 or x == 4):
            ADR_signal_MHRA = 1
            #print("PRR_X有測出不良反應")

        #SPRT
        try:
            sprt_e = (a+b) * (a+c) / (a+b+c+d)
            sprt = math.log(2) * a - sprt_e
            #print("SPRT = " + str(sprt))
        except ZeroDivisionError as e: #處理異常
            sprt = 0
        
        if sprt > 2.93:
            ADR_signal_SPRT = 1
            #print("SPRT有測出不良反應")

        #Yule'S Q
        try:
            q = ((a*d) - (b*c)) / ((a*d) + (b*c))
            #print("Q = " + str(q))
            se_q = math.sqrt((1/a) + (1/b) + (1/c) + (1/d))
            #print("SE_Q = " + str(se_q))
            try:
                ci_q = q - 1.96 * ((1/2) * (1-pow(q,2)) * se_q)
            except ZeroDivisionError:
                ci_q = -1 
                #ci_q = 0 
            #print("CI_Q = " + str(ci_q))
        except ZeroDivisionError as e: #處理異常
            ci_q = 0 
        
        if ci_q > 0:
            ADR_signal_Q = 1
            #print("Q有測出不良反應")
            
        return (ADR_signal_ROR,
        ADR_signal_PRR,
        ADR_signal_BCPNN,
        ADR_signal_MHRA,
        ADR_signal_SPRT,
        ADR_signal_Q)
class counter():
    TP,TN,FP,FN = 0,0,0,0
#========================================
cror=counter()
cprr=counter()
cbcpnn=counter()
cmhra=counter()
csprt=counter()
cq=counter()
while point:
    df = reader.get_chunk(steps)
    data_list = df.values.astype(int)
    for i in range(0,len(data_list)):
       a = data_list[i][76]
       b = data_list[i][77]
       c = data_list[i][78]
       d = data_list[i][79]
       true_label = data_list[i][80]
       #print(data_list[i])
       #print(a,b,c,d,true_label)
       
       ror,prr,bcpnn,mhra,sprt,q = ADRs(a,b,c,d)
       #print(ror,prr,bcpnn,mhra,sprt,q)
       
       if ror == 0:
            if true_label == 0:
                cror.TN+=1
            else:   #label = 1
                cror.FN+=1
       else:
            if true_label == 1:
                cror.TP+=1
            else:   #label = 1
                cror.FP+=1
       #print(cror.TP,cror.TN,cror.FP,cror.FN)

       if prr == 0:
            if true_label == 0:
                cprr.TN+=1
            else:   #label = 1
                cprr.FN+=1
       else:
            if true_label == 1:
                cprr.TP+=1
            else:   #label = 1
                cprr.FP+=1
       #print(cprr.TP,cprr.TN,cprr.FP,cprr.FN)

       if bcpnn == 0:
            if true_label == 0:
                cbcpnn.TN+=1
            else:   #label = 1
                cbcpnn.FN+=1
       else:
            if true_label == 1:
                cbcpnn.TP+=1
            else:   #label = 1
                cbcpnn.FP+=1
       #print(cbcpnn.TP,cbcpnn.TN,cbcpnn.FP,cbcpnn.FN)

       if mhra == 0:
            if true_label == 0:
                cmhra.TN+=1
            else:   #label = 1
                cmhra.FN+=1
       else:
            if true_label == 1:
                cmhra.TP+=1
            else:   #label = 1
                cmhra.FP+=1
       #print(cmhra.TP,cmhra.TN,cmhra.FP,cmhra.FN)

       if sprt == 0:
            if true_label == 0:
                csprt.TN+=1
            else:   #label = 1
                csprt.FN+=1
       else:
            if true_label == 1:
                csprt.TP+=1
            else:   #label = 1
                csprt.FP+=1
       #print(csprt.TP,csprt.TN,csprt.FP,csprt.FN)

       if q == 0:
            if true_label == 0:
                cq.TN+=1
            else:   #label = 1
                cq.FN+=1
       else:
            if true_label == 1:
                cq.TP+=1
            else:   #label = 1
                cq.FP+=1
       #print(cq.TP,cq.TN,cq.FP,cq.FN)

       #print("======")
       running+=1

       
    if running >= data_num:
        end = time.process_time()
        print("RunTime : %f s" % (end - start))
        print("Counter :" ,running,"/",data_num)
        point = False
    else:
        print("Testing:",running)

print("=======================================")
TP, TN, FP, FN = cror.TP,cror.TN,cror.FP,cror.FN
print("The validation index of ROR:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))

print("=======================================")
TP, TN, FP, FN = cprr.TP,cprr.TN,cprr.FP,cprr.FN
print("The validation index of PRR:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))

print("=======================================")
TP, TN, FP, FN = cbcpnn.TP,cbcpnn.TN,cbcpnn.FP,cbcpnn.FN
print("The validation index of BCPNN:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))

print("=======================================")
TP, TN, FP, FN = cmhra.TP,cmhra.TN,cmhra.FP,cmhra.FN
print("The validation index of MHRA:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))

print("=======================================")
TP, TN, FP, FN = csprt.TP,csprt.TN,csprt.FP,csprt.FN
print("The validation index of SPRT:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))

print("=======================================")
TP, TN, FP, FN = cq.TP,cq.TN,cq.FP,cq.FN
print("The validation index of Yule's Q:")
print("Overall Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),2))
print("Average Accuracy: ",round(((TP/(TP+FN))+(TN/(TN+FP)))/2,2))
#print("True Positive Rate : ",round((TP/(TP+FN)),4))
print("Specificity : ",round((TN/(TN+FP)),2))
print("Precision : ",round((TP/(TP+FP)),2))
print("Recall : ",round((TP/(TP+FN)),2))
print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),2))


print("=======================================")
winsound.Beep(2000,1000)
print("All Done.")

