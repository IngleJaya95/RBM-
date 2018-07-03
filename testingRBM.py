
# coding: utf-8

# In[120]:


import numpy as np 
import pandas as pd 
import sys 
import os 
import pickle 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# In[121]:


def saveObject(obj,name): 
    pickle.dump(obj,open( "tsne/"+name+".pkl", "wb" ))

def loadObject(name):
    obj = pickle.load( open( "tsne/"+name+".pkl", "rb" ) )
    return obj


# In[122]:


BintestFeatures = loadObject("BintestFeatures")
BintestLabels = loadObject("BintestLabels")


# In[123]:




(test_row,test_col) = (np.shape(BintestFeatures)) #Validation Operation
(t_row) = (np.shape(BintestLabels)) #Validation Operation


# In[124]:


#BintestFeatures[0]


# In[125]:


np.random.seed(1234)
k = 1
num_hidden = 3
num_visible = 784
lr = 0.001
epoch = 0


# In[126]:


np_rng = np.random.RandomState(1234)


# In[127]:


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def test(point,W,b,c):
    hid = sigmoid(np.dot(point,W) + b)
    re = sigmoid(np.dot(hid,W.T) + c)
    return re


# # MAiN CODE

# In[128]:


print("Here the testing starts, please check k and epoch ")

[W,b,c] = loadObject(str(num_hidden))
abstract = test(BintestFeatures,W,b,c)
saveObject(abstract,'abstract_hid_'+str(num_hidden))


# In[129]:


#Extract data

# abstract = np.zeros([points,dim])
# abstract = loadObject('abstract_hid_'+str(num_hidden)+'_itr_'+str(epoch)+'_k_'+str(k)+'_lr_'+str(lr))


# In[130]:


#Extract label
# BintestLabels = np.zeros([points,1])
# BintestLabels = loadObject('BintrainLabels')


# In[131]:


feat_cols = [ 'pixel'+str(i) for i in range(784) ]

df = pd.DataFrame(abstract,columns=feat_cols)
df['label'] = BintestLabels
df['label'] = df['label'].apply(lambda i: str(i))


# In[144]:


rndperm = np.random.permutation(df.shape[0])

n_sne = 10000 # since we have 10000 test images 

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=250,learning_rate=100.0)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]


saveObject(df_tsne,'tsnehid_'+str(num_hidden))


# In[145]:


from ggplot import *


# In[147]:


chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') )         + geom_point(size=70,alpha=0.1)         + ggtitle("k = "+str(num_hidden)+" tsne")
chart

chart.save(str(num_hidden)+"32"+".pdf")


# In[38]:




