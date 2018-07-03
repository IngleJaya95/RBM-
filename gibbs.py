
# coding: utf-8

# In[36]:


import numpy as np 
import pandas as pd 
import sys 
import os 
import pickle 
from sklearn.metrics import mean_squared_error


# In[15]:


np.random.seed(1234)
train_epochs = 1
k = 1
num_hidden = 123
num_visible = 784
lr = 0.001
#num_hidden_setter = [400]
location_log_file = ""
training = 1 # 1 for training and 0 for testing
uniqueName = ""
OtherName =location_log_file +uniqueName + "hiddenUnit-" + str(num_hidden) + "-K-" + str(k) + "-lr-" + str(lr) + ".csv"

logFile = open(OtherName,'w')

logFile.write("iteration,loss")
patienceParameter = 5

lossStep = 1000 # After how many steps you want to see


# In[16]:


def saveObject(obj,name): 
    pickle.dump(obj,open( "repo/"+name+".pkl", "wb" ))

def loadObject(name):
    obj = pickle.load( open( "repo/"+name+".pkl", "rb" ) )
    return obj


# In[17]:


BintrainFeatures = loadObject("BintrainFeatures")
BintrainLabels = loadObject("BintrainLabels")


# In[18]:


(no_of_row,no_of_col) = (np.shape(BintrainFeatures)) #Training Operation
l_row = (np.shape(BintrainLabels)) #Training Labels


# In[19]:


np_rng = np.random.RandomState(1234)
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
def hGivenv(pt,W,b): #will give h
        Mean_h = sigmoid(np.dot(pt,W) + b)
        Sample_h = np_rng.binomial(size = Mean_h.shape,n = 1,p = Mean_h)
        #print(h1_sample)
        return [Mean_h, Sample_h]
def vGivenh(h,W,c):# will give v 
        Mean_v = sigmoid(np.dot(h,W.T) + c)
        Sample_v = np_rng.binomial(size = Mean_v.shape, n=1,p = Mean_v)
        
        return [Mean_v, Sample_v]
def gibbs_chain(h,W,b,c):
        Mean_v, Sample_v = vGivenh(h,W,c)
        Mean_h, Sample_h = hGivenv(Sample_v,W,b)

        return [Mean_v, Sample_v,Mean_h,Sample_h]
def loss_ce(point,W,b,c):
    
        hidden_sigmoid = sigmoid(np.dot(point,W) + b)
        visible_sigmoid = sigmoid(np.dot(hidden_sigmoid, W.T) + c)
        sum1  = np.sum(point * np.log(visible_sigmoid) +(1 - point) * np.log(1 - visible_sigmoid))
        loss =  - np.mean(sum1)
        
        return loss


# In[76]:


minLossPrev = 1000000000000000
chanceCount = 0


W = np.random.randn(num_visible,num_hidden)/np.sqrt(num_visible/2)
b = np.zeros([num_hidden])
c = np.zeros([num_visible])
cost = np.zeros([train_epochs,no_of_row])


# try:

avgLoss = 0
iterationLoss = 0
print("Here the training starts")

for epoch in range(train_epochs):
    #print(epoch)
    for i in range(10):

        point = np.array([BintrainFeatures[i-1,:]])     #[1,784]
        Sample_v  = np_rng.binomial(size = point.shape,n = 1,p = 0.01)         
        #Sample_v = np.reshape(Sample_v,[1,num_visible])  #[1,784]
        Mean_h, Sample_h = hGivenv(Sample_v,W,b)
        Mean_h = np.reshape(Mean_h,[1,num_hidden] )       #[1,123]
        #chain starts at current sample

        step = 0
        find = 0 
        point2 = np.reshape(point,[1,num_visible])
        samplev2 = np.reshape(Sample_v,[1,num_visible])
        error = mean_squared_error(point2,samplev2 )
        print("error ",error)
        while(error > 0.001):
            print("find",find)
            find = find+1
            if step == 0:
                Mean_v_new, Sample_v_new,Mean_h_new,Sample_h_new= gibbs_chain(Sample_h,W,b,c)
#                 prevsample = Sample_v_new
                #print(Sample_v_new)
                step = 1
            else:
                Mean_v_new, Sample_v_new,Mean_h_new, Sample_h_new = gibbs_chain(Sample_h_new,W,b,c)
                
                
            point2 = np.reshape(point,[1,num_visible])
            samplev2 = np.reshape(Sample_v_new,[1,num_visible])
            
            error = mean_squared_error(point2 , samplev2)
            print(error)
            
        print("samples drawn", find,"at itr",epoch,"i = ",i)
        # negative sample is Sample_v_new
        #print(np.shape(np.array(point.T)))
        #print(np.shape(Mean_h))
        svn = np.reshape(Sample_v_new,[1,num_visible])
        mh = np.reshape(Mean_h_new,[1,num_hidden])
        #print(np.shape(svn))
        #print(np.shape(mh))
        W = W + (lr * ((np.dot(point.T, Mean_h) - np.dot(svn.T, mh))))
        b = b + (lr * (np.mean(Mean_h - mh, axis=0) )) #hbias = b
        c = c + (lr * np.mean(point - svn, axis=0))

        avgLoss += loss_ce(point,W,b,c)
        iterationLoss += loss_ce(point,W,b,c)
        if i%lossStep == 0:
            normAvgLoss = avgLoss/lossStep
            #print(epoch,i,normAvgLoss)
            avgLoss = 0

    avgIterationLoss = iterationLoss/no_of_row
    #print("\n",avgIterationLoss,"\n")
    iterationLoss = 0
    logFile.write(str(epoch)+","+str(avgIterationLoss))

    if(minLossPrev > avgIterationLoss):
        minLossPrev = avgIterationLoss
        chanceCount = 0
    else:
        chanceCount += 1
        if chanceCount >= patienceParameter:
                print("You are done Training :)")
                saveObject([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr)) 
                sys.exit()
    #need to save the model at the end 
    saveObject([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr))

# except:

#     print("Some Error occured, Saving your lifeline : Model :)")
#     saveObject([W,b,c],'hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr))
    


# In[ ]:


Mean_h, Sample_h = hGivenv(Sample_v,W,b)
Mean_h = np.array(Mean_h)
Mean_h =np.reshape(Mean_h,[num_hidden,1])
np.shape(Mean_h)


# In[61]:


np.dot(point.T, Mean_h)


# In[42]:


y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(point, Sample_v)


# In[31]:


np.shape(Mean_h
        )


# In[35]:


a = [1,2,3]
b = [1,2,903]


# In[36]:


x = (a ==b)


# In[37]:


if x is True:
    print("1")


# In[5]:


x = np.random.rand(num_visible)


# In[20]:


Sample_v  = np.random.rand(num_visible)


# In[21]:


np.shape(Sample_v)


# In[70]:


a = 2
b = 3


# In[71]:


while(a<b):
    a = a+0.5
    print(a)

