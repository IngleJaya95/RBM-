
# coding: utf-8

# In[1]:


import numpy as np 
#import pandas as pd 
import sys 
import os 
import pickle 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


import signal
import sys


# Hyperparamters Set

# In[118]:


np.random.seed(1234)
train_epochs = 100
k = 1
num_hidden = 150
num_visible = 784
lr = 0.001
#num_hidden_setter = [400]
location_log_file = ""
training = 1 # 1 for training and 0 for testing
uniqueName = "150hid_"

if not os.path.exists("repo/"+uniqueName):
    os.makedirs("repo/"+uniqueName)

locaa = "repo/" + uniqueName + "/" 
OtherName = locaa +uniqueName + "hiddenUnit-" + str(num_hidden) + "-K-" + str(k) + "-lr-" + str(lr) + ".csv"

logFile = open(OtherName,'w')

logFile.write("iteration,loss")
patienceParameter = 5

lossStep = 1000 # After how many steps you want to see


# # Data loading steps

# In[120]:


# datasetLoc = "repo"
# trainD = pd.read_csv( datasetLoc + "Bintrain.csv", low_memory=False,header=None)
# trainL = pd.read_csv( datasetLoc + "BintrainLabels.csv", low_memory=False,header=None)
# #testD = pd.read_csv( datasetLoc + "Bintest.csv", low_memory=False,header=None)
# #testL = pd.read_csv( datasetLoc + "BintestLabels.csv", low_memory=False,header=None)

#since we dont have rightnow


# In[122]:


def saveObject(obj,name): 
    pickle.dump(obj,open( "repo/core/"+name+".pkl", "wb" ))

def loadObject(name):
    obj = pickle.load( open( "repo/core/"+name+".pkl", "rb" ) )
    return obj

def saveObjectMod(obj,name): 
    pickle.dump(obj,open( locaa+name+".pkl", "wb" ))

def loadObjectMod(name):
    obj = pickle.load( open( locaa+name+".pkl", "rb" ) )
    return obj


# In[127]:


# BintrainFeatures = np.zeros([60000,784],dtype =int)  
# BintrainLabels  = np.zeros([60000,1],dtype = int)
# # change it accordingly 
# BintestFeatures = np.zeros([60000,784],dtype =  int)
# BintestLabels = np.zeros([60000,784],dtype =  int)


# In[145]:


# #training
# trainDnp = trainD.values
# BintrainFeatures = trainDnp[0:60000,0:785] #Traning Input data


# # labels
# trainLnp = trainL.values
# BintrainLabels = trainLnp[0:60000,0:785] #Traning Input data


#####Check here starts 
#Testing
# testDnp = trainD.values   
# BintestFeatures = testDnp[0:60000,0:785] #Testing we don't have Labels we are using training data here 

# testLnp = trainL.values
# BintestLabels = testLnp[0:60000,0:785] #Testing we don't have Labels we are using training data here 
#####Check here ends

BintrainFeatures = loadObject("BintrainFeatures")
BintrainLabels = loadObject("BintrainLabels")


# In[151]:


(no_of_row,no_of_col) = (np.shape(BintrainFeatures)) #Training Operation
l_row = (np.shape(BintrainLabels)) #Training Labels


# In[152]:


#no_of_row



# # Support Functions

# In[153]:


np_rng = np.random.RandomState(1234)


# In[154]:


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# In[155]:


def hGivenv(pt,W,b): #will give h
        Mean_h = sigmoid(np.dot(pt,W) + b)
        Sample_h = np_rng.binomial(size = Mean_h.shape,n = 1,p = Mean_h)
        #print(h1_sample)
        return [Mean_h, Sample_h]


# In[156]:


def vGivenh(h,W,c):# will give v 
        Mean_v = sigmoid(np.dot(h,W.T) + c)
        Sample_v = np_rng.binomial(size = Mean_v.shape, n=1,p = Mean_v)
        
        return [Mean_v, Sample_v]


# In[157]:


def gchain(h,W,b,c):
        Mean_v, Sample_v = vGivenh(h,W,c)
        Mean_h, Sample_h = hGivenv(Sample_v,W,b)

        return [Mean_v, Sample_v,Mean_h,Sample_h]
    


# In[158]:


def loss_ce(point,W,b,c):
    
        hidden_sigmoid = sigmoid(np.dot(point,W) + b)
        visible_sigmoid = sigmoid(np.dot(hidden_sigmoid, W.T) + c)
        sum1  = np.sum(point * np.log(visible_sigmoid) +(1 - point) * np.log(1 - visible_sigmoid))
        loss =  - np.mean(sum1)
        
        return loss


# # initilization

# # MAIN CODE

# In[160]:




minLossPrev = 1000000000000000
chanceCount = 0


W = np.random.randn(num_visible,num_hidden)/np.sqrt(num_visible/2)
b = np.zeros([num_hidden])
c = np.zeros([num_visible])
cost = np.zeros([train_epochs,no_of_row])

[W,b,c] = loadObjectMod("hid_150_itr_19_k_1_lr_0.001")

startE = 0
try:

    avgLoss = 0
    iterationLoss = 0
    print("Here the training starts")
    
    for epoch in range(startE,train_epochs):
        print(epoch)
        for i in range(no_of_row):
            
            point = np.array([BintrainFeatures[i-1,:]])
            Mean_h, Sample_h = hGivenv(point,W,b)
            
            #chain starts at current sample

            for step in range(k):
                    if step == 0:
                        Mean_v_new, Sample_v_new,Mean_h_new,Sample_h_new= gchain(Sample_h,W,b,c)
                    else:
                        Mean_v_new, Sample_v_new,Mean_h_new, Sample_h_new = gchain(Sample_h_new,W,b,c)

            # negative sample is Sample_v_new

            W = W + (lr * ((np.dot(point.T, Mean_h) - np.dot(Sample_v_new.T, Mean_h_new))))
            b = b + (lr * np.mean(Mean_h - Mean_h_new, axis=0) ) #hbias = b
            c = c + (lr * np.mean(point - Sample_v_new, axis=0))
            
            avgLoss += loss_ce(point,W,b,c)
            iterationLoss += loss_ce(point,W,b,c)
            if i%lossStep == 0:
                normAvgLoss = avgLoss/lossStep
                print(epoch,i,normAvgLoss)
                avgLoss = 0
                
        avgIterationLoss = iterationLoss/no_of_row
        print("\n",avgIterationLoss,"\n")
        iterationLoss = 0
        logFile.write(str(epoch)+","+str(avgIterationLoss))

        if(minLossPrev > avgIterationLoss):
            minLossPrev = avgIterationLoss
            chanceCount = 0
        else:
            chanceCount += 1
            if chanceCount >= patienceParameter:
                    print("You are done Training :)")
                    saveObjectMod([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr)) 
                    sys.exit()
        #need to save the model at the end 
        saveObjectMod([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr))

except:

    print("Some Error occured, Saving your lifeline : Model :)")
    saveObjectMod([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr))
    
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        saveObjectMod([W,b,c],'hid_'+str(num_hidden)+'_itr_'+"quit"+'_k_'+str(k)+'_lr_'+str(lr))
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')
signal.pause()



# # For TESTING 

# # Plotting the representation

# In[64]:


# ppp = df_tsne.values
# x = ppp[:,785]
# y = ppp[:,786]
# label = ppp[:,784]
# label = label.astype(float)
# label = label.astype(int)
# xlist = []
# ylist = []
# cc = [[1,1,0],[0,1,1],[0,0,1],[0,1,0],[1,0,0],[1,0,1],[0,0.5,0.5],[0.5,1,0.2],[0.3,0.2,0.1],[0.5,0.8,0.234]]
# for i in range(10):
#     xp = x[label == i]
#     yn = y[label == i]
#     plt.scatter(xp,yn,c=cc[i])

# #plt.scatter(x,y)
# #x1.shape


# In[212]:


# x = ppp[:,785]
# y = ppp[:,786]
# label = ppp[:,784]
# label = label.astype(float)
# label = label.astype(int)
# xlist = []
# ylist = []
# cc = [[1,1,0],[0,1,1],[0,0,1],[0,1,0],[1,0,0],[1,0,1],[0,0.5,0.5],[0.5,1,0.2],[0.3,0.2,0.1],[0.5,0.8,0.234]]
# for i in range(10):
#     xp = x[label == i]
#     yn = y[label == i]
#     plt.scatter(xp,yn,c=cc[i])

# #plt.scatter(x,y)
# #x1.shape

