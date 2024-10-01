import csv
import os
import gc
import time
import math
import random
import winsound
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
np.seterr(divide='ignore', invalid='ignore')
random.seed(2)
np.random.seed(2)
test_need = 4791223
reader   = pd.read_csv('./data/ATCPTLabeled_OHE_Test_PU_'+str(test_need)+'.csv', header=None, iterator=True)
print("Iteration loaded")
#------------------------------------------------------------------------------------------
steps = 10000
#------------------------------------------------------------------------------------------
#Train DATA Generator
try:
    os.remove('./data/ATCPTLabeled_PU_Test.csv')
except OSError:
    print("")

point = True
start = time.process_time()
Pflag=0
Uflag=0
Oflag=0
check = -1
while point:
    try:
        df = reader.get_chunk(steps)
        check+=1
        data = pd.DataFrame()
        for i in range(0,len(df)):
            a = df.iloc[i][76]
            b = df.iloc[i][77]
            c = df.iloc[i][78]
            d = df.iloc[i][79]
            N = a + b + c + d
            #print("------------------------------------------------------------------------")
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
                bcp_e = math.log(bcp_e_ic,2)
                #print("IC = " + str(ic))
            except ZeroDivisionError as e: #處理異常
                bcp_e = 0
                
            try:
                bcp_sd_a = (N-a-b+1) / ((a+b+1) * (N+3))
                bcp_sd_b = (N-a-c+1) / ((a+c+1) * (N+3))
                bcp_sd_r = (N-a+bcp_r-1) / ((a+1) * (1+N+bcp_r))
                bcp_sd = math.sqrt(bcp_sd_a + bcp_sd_b + bcp_sd_r)
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
                
            ADrs = ( ADR_signal_ROR*1+
            ADR_signal_PRR*1+
            ADR_signal_BCPNN*1+
            ADR_signal_MHRA *1+
            ADR_signal_SPRT*1+
            ADR_signal_Q *1)
            #print("------------------------------------------------------------------------")
            if ADrs <=1 :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Uflag+=1
            elif ADrs >=5 :
                data = pd.concat([data,df.iloc[[i]]],axis=0)
                Pflag+=1
            else :
                Oflag+=1
        csvPath = './data/ATCPTLabeled_PU_Test.csv'
        data.to_csv(csvPath,header=False,index=None,mode='a')
        print("Test Data not enough ! ",  "P=",Pflag,"U=", Uflag,"All=", (Pflag+Uflag+Oflag),"/4102229")

    except StopIteration:
        point = False
        end = time.process_time()
        print("Done")
        print("Count : ",  "P=",Pflag,"U=", Uflag)
        print(Pflag / (Pflag+Uflag),"P:U", Uflag / (Pflag+Uflag))
        
print(" ")

