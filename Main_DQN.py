import numpy as np
import pandas as pd
import gc
import gym
import os
import random
import math
import time
import winsound
from gym import spaces
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from tensorflow.keras import regularizers
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) # this goes *before* tf import
import tensorflow as tf
from sklearn import metrics
tf.get_logger().setLevel('ERROR')
np.seterr(divide='ignore', invalid='ignore')
random.seed(2)
np.random.seed(2)

#"------------------------------------------------------------------------

#建立強化學習環境
class CustomTabularEnv(gym.Env):
    def __init__(self,dataset: np.ndarray):
        self.state_space = dataset # example state space
        self.action_space = spaces.Discrete(2) # example action space
        self.current_state = None #初始化
        self.m,self.n=dataset.shape # 取陣列之長與寬
        self.m_feature=self.m-1
        self.n_feature=self.n-1
        #self.n_samples=self.m
        self.x=dataset[:,:self.n_feature]
        self.y=dataset[:,self.n_feature]
        #For Confusion Matrix
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.PredictLabel = np.empty(0)
        self.TrueLabel = np.empty(0)

    def reset(self):
        self.counts = 0 #初始化計算器
        self.random_num = np.random.choice(self.m_feature, replace=True)
        self.current_state = self.x[self.random_num]
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        return self.current_state
    
    def step(self, action):
        self.counts += 1
        reward = self.calculate_reward1(self.current_state, action) # calculate reward based on current state and action
        self.TrueLabel=np.append(self.TrueLabel,self.y[self.random_num])
        self.random_num = np.random.choice(self.m_feature, replace=True)
        if self.counts>=200: # 一輪運行x次
          done = True
          #self.random_num -=1
        else:
          done = False # set done flag to False
        next_state = self.x[self.random_num] # generate next state
        self.current_state = next_state # update current state
        info = {}
        return next_state, reward, done, info
    
    def calculate_reward(self, state, action):
        Adr = self.ADRs(state)
        if action == 1 :
            self.PredictLabel=np.append(self.PredictLabel,1)
            if self.y[self.random_num] == 1:
                self.TP += 1
                return 1
            elif Adr >=(4/6):
                self.y[self.random_num] = 1
                self.TP += 1
                return 1
            else:
                self.FP += 1
                return -1+Adr
        elif action == 0 :
            self.PredictLabel=np.append(self.PredictLabel,0)
            if self.y[self.random_num] == 0:
                self.TN += 1
                return 1
            elif Adr <=(1/6):
                self.TN += 1
                return 1
            else :
                self.FN += 1
                return -1


    def calculate_reward1(self, state, action):
        Adr = self.ADRs(state)
        #if Adr >=4:
            #self.y[self.random_num] = 1
        if action == 1 :
            self.PredictLabel=np.append(self.PredictLabel,1)
            if self.y[self.random_num] == 1:
                self.TP += 1
                return 1
            elif Adr ==(6/6):
                self.y[self.random_num] = 1
                self.TP += 1
                return 0.8
            elif Adr ==(5/6):
                self.y[self.random_num] = 1
                self.TP += 1
                return 0.7
            elif Adr ==(4/6):
                self.y[self.random_num] = 1
                self.TP += 1
                return 0.6
            else:
                self.FP += 1
                return -1+Adr
        elif action == 0 :
            self.PredictLabel=np.append(self.PredictLabel,0)
            if self.y[self.random_num] == 0:
                self.TN += 1
                return 1
            elif Adr <=(1/6):
                self.TN += 1
                return 0.8
            elif Adr ==(2/6):
                self.TN += 1
                return 0.7
            elif Adr ==(3/6):
                self.TN += 1
                return 0.6
            else :
                self.FN += 1
                return -Adr

    def calculate_reward66(self, state, action):
        if action == 1 :
            self.PredictLabel=np.append(self.PredictLabel,1)
            if self.y[self.random_num] == 1:
                self.TP += 1
                return 1
            else:
                self.FP += 1
                return -1
        elif action == 0 :
            self.PredictLabel=np.append(self.PredictLabel,0)
            if self.y[self.random_num] == 0:
                self.TN += 1
                return 1
            else :
                self.FN += 1
                return -1


    def PrintTheSpace(self):
        return self.state_space
    
    def PrintConfusionMatrix(self):
        return self.TP, self.TN, self.FP, self.FN

    def ADRs(self, state):
        
        a = state[self.n_feature-4]
        b = state[self.n_feature-3]
        c = state[self.n_feature-2]
        d = state[self.n_feature-1]
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
            
        return (( ADR_signal_ROR*1+
        ADR_signal_PRR*1+
        ADR_signal_BCPNN*1+
        ADR_signal_MHRA *1+
        ADR_signal_SPRT*1+
        ADR_signal_Q *1)/6)
    
#"------------------------------------------------------------------------
'''
#測試用
    
#env=CustomTabularEnv(dataset=inputdata)
#env.PrintTheSpace()[0]
'''
#"------------------------------------------------------------------------
data = pd.read_csv('.\data\ATCPTLabeled_OHE_reset.csv', header=None)
inputdata = data.values
#釋放記憶體
del data
gc.collect()
#"------------------------------------------------------------------------
'''
#測試環境用
#通過迴圈輸出觀察函數，並在action_space隨機抽取一個值作action，再傳入至環境換取下一個值。
env=CustomTabularEnv(dataset=inputdata)
print("-測試環境----------------------------------------------------------------")
print("------------------------------------------------------------------------")

for i_episode in range(1): #how many episodes you want to run
  observation = env.reset() #reset() returns initial observation
  for t in range(10):
    print("----------------------------------------------")
    print(observation.flatten())
    #action = env.action_space.sample()
    action = 0
    observation, reward, done, info = env.step(action)
    print("Action : ",action)
    print("Label : ",info)
    print("Reward : ",reward)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
'''
#"------------------------------------------------------------------------
#引用環境
env=CustomTabularEnv(dataset=inputdata)
nb_actions = env.action_space.n
#建立神經網絡模型
model = Sequential()

model.add(Flatten(input_shape=(1,(env.n_feature))))
model.add(Dense(12,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
    
print(model.summary())
#"------------------------------------------------------------------------
#宣告變數
ENV_NAME = "ADRs"
#here
binary = 2

if binary ==0:
    Train = 1
    Test = 1
    Test_inf = 0
if binary ==1:
    Train = 0
    Test = 1
    Test_inf = 1
if binary ==2:
    Train = 1
    Test = 1
    Test_inf = 1
if binary ==3:
    Train = 0
    Test = 0
    Test_inf = 0
    
nb_steps_set = 200000#2000000
nb_steps_warmup_set = 10000 #50000
memory_size = 200000 #100000
target_update = memory_size/10 
#"------------------------------------------------------------------------
#建立規則
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=nb_steps_set)
memory = SequentialMemory(limit = memory_size, window_length = 1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup_set,batch_size=128,
target_model_update=target_update, policy=policy, enable_double_dqn=False)#,enable_dueling_network = False,dueling_type='avg')

dqn.compile(optimizer=Adam(learning_rate=3e-4),metrics=['mae'])
#"------------------------------------------------------------------------
if Train :
    data = pd.read_csv('.\data\ATCPTLabeled_OneHotEncoder_ForATC.csv', header=None)
    inputdata = data.values
    #釋放記憶體
    del data
    gc.collect()
    env=CustomTabularEnv(dataset=inputdata)
    #訓練模型
    print("Train !")
    dqn.fit(env, nb_steps = (nb_steps_set) , verbose=2)
    print("Done !")

#"------------------------------------------------------------------------
#保存模型權重
    dqn.save_weights("dqn_ADRs_weights", overwrite=True)
#"------------------------------------------------------------------------

if Test :
    # Load testdata
    #testdata = pd.read_csv('.\data\ATCPTLabeled_OHE_Test_Balance_300000.csv', header=None)
    #testdata = pd.read_csv('.\data\ATCPTLabeled_OHE_Test_PU_2019_01-03_Clean.csv', header=None)
    #testdata = pd.read_csv('.\data\ATCPTLabeled_OHE_Test_PU_2019_01-03.csv', header=None)
    testdata = pd.read_csv('.\data\TEST2019Q1_Clean.csv', header=None)

    #轉換為陣列格式
    test_dataset = testdata.values
    test_X, test_y=test_dataset[:,:-1], test_dataset[:,-1]
    #釋放記憶體
    del testdata
    gc.collect()
    

    env=CustomTabularEnv(dataset=test_dataset)
    try:
        dqn.load_weights("dqn_ADRs_weights")
        print("Test !")
        loop = True
        #How many round :
        Test_episode = 500
        ATP= 0
        ATN= 0
        AFP= 0
        AFN = 0
        for k in  range(Test_episode):
            dqn.test(env, nb_episodes=1, visualize=False)
            print("=======================================")
            #Confusion Matrix
            if Test_inf :
                TP, TN, FP, FN = env.PrintConfusionMatrix()
                ATP=ATP+TP
                ATN=ATN+TN
                AFP=AFP+FP
                AFN=AFN+FN
                print("TP : ",TP, "TN : ",TN, "FP : ",FP, "FN : ",FN )
                if TP != 0 and TN != 0 and FP != 0 and FN != 0 :
                    print("Accuracy : ",round(((TP+TN)/(TP+TN+FP+FN)),4))
                    print("Precision : ",round((TP/(TP+FP)),4))
                    print("Recall : ",round((TP/(TP+FN)),4))
                    print("F1 : ",round((2/((1/(TP/(TP+FP)))+(1/(TP/(TP+FN))))),4))
                    print("=======================================")

                if k == Test_episode-1:
                    print("=======================================")
                    print("Summary:")    
                    print("TP : ",ATP, "TN : ",ATN, "FP : ",AFP, "FN : ",AFN,"|| P:",(ATP+AFN),"U:",(ATN+AFP))
                    Aprecision = metrics.precision_score(env.TrueLabel, env.PredictLabel,average='weighted')
                    Arecall = metrics.recall_score(env.TrueLabel,env.PredictLabel,average='weighted')
                    Af = metrics.fbeta_score(env.TrueLabel, env.PredictLabel,average='weighted',beta=1)
                    if ATP != 0 and ATN != 0 and AFP != 0 and AFN != 0 :
                        print("Overall Accuracy : ",round(((ATP+ATN)/(ATP+ATN+AFP+AFN)),4))
                        print("Average Accuracy : ",round(((ATP/(ATP+AFN))+(ATN/(ATN+AFP)))/2,4))
                        print("True Positive Rate : ",round((ATP/(ATP+AFN)),4))
                        print("True Negative Rate : ",round((ATN/(ATN+AFP)),4))
                        print("Precision : ",round(Aprecision,4))
                        print("Recall : ",round(Arecall,4))
                        print("F1 : ",round(Af,4))
                    np.savetxt(".\data\PredictLabelsFromDQN.csv", env.PredictLabel, fmt='%i', delimiter=",")
                    winsound.Beep(2000,1000)
    except OSError:
        print('Could not find model file. Continue')
        
        
    print("All Done")
#"------------------------------------------------------------------------