
# coding: utf-8

# In[1]:


#Author: Jinwei Wang
#Date: 8-6-2018

import numpy as np


# In[2]:


#Create Data
x1=np.array([29,43,50,36,2,29,94,45,83,93,41,76,39,91,45,28,28,39,57,
    72,12,38,52,83,32,15,100,49,83,93,50,45,72,56,36,43,11,
    23,67,67,2,86,14,92,51,52,94,55,87,40,14,30,24,22,17,22,
    7,43,71,23,32,33,53,32,27,94,19,54,76,50,67,53,2,45,73,98,
    19,15,9,24,64,51,62,86,58,32,53,45,52,85,84,29,83,67,91,45,71,8,76,99]);  
x2=np.array([54,47,65,79,56,80,55,65,85,67,67,42,45,56,21,67,72,50,33,2,
    81,55,66,13,30,63,55,7,39,19,61,43,35,69,45,69,39,83,50,60,
    44,10,56,99,59,17,66,68,41,83,26,42,79,0,14,48,44,96,65,37,
    61,85,72,40,24,82,0,44,58,35,43,28,67,58,37,28,34,40,61,45,
    22,28,60,91,81,51,83,75,97,91,58,83,68,57,55,38,87,59,100,80]);
x3=np.array([98,56,31,78,30,90,73,52,37,21,93,97,25,60,90,66,28,69,46,67,
    32,56,95,6,1,86,52,89,61,26,82,97,52,43,39,95,59,2,22,6,83,
    91,0,51,76,94,45,37,11,40,9,12,7,19,60,38,17,76,55,89,91,44,
    18,55,24,73,32,29,75,9,69,95,84,69,58,59,93,37,1,57,84,75,17,
    70,40,6,86,11,71,64,95,19,21,17,3,79,33,41,19,42]);  
y1=np.array([973,202,77,541,59,796,429,187,131,63,853,938,40,256,738,335,77,357,
    114,308,100,210,906,10,12,677,181,710,250,30,594,936,160,133,83,909,
    222,71,42,43,591,763,33,240,479,839,144,102,27,137,9,22,65,9,220,80,
    25,535,216,721,794,161,63,186,22,466,35,49,463,18,354,871,638,367,216,
    223,818,68,38,208,604,435,47,434,135,29,710,62,457,353,899,79,64,44,
    39,512,119,105,114,148]);  
y2=np.array([63,107,170,117,34,97,868,139,648,851,123,466,82,791,105,73,77,91,201,380,
    71,91,194,574,42,52,1035,127,593,811,170,119,391,228,71,137,22,81,
    328,337,28,646,34,882,175,153,879,216,676,137,10,46,77,13,13,37,21,179,
    406,35,79,113,203,54,28,905,10,180,480,138,326,166,53,132,409,955,28,23,
    38,40,275,148,276,726,265,59,226,148,242,703,636,95,620,335,784,113,
    437,39,541,1038]);  


# In[3]:


#Create Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[4]:


#Create the first derivative of the Sigmoid function
def sigderivative(x):
    return x*(1-x)


# In[5]:


#Transfer y1&y2 into sigmoid
s1 = sigmoid(y1)
s2 = sigmoid(y2)


# In[6]:


def Backpro(training_sample,learnrate,epochs):
    #Feedforward

    #Take the whole sample
    m,n=training_sample.shape
    #Get the X values
    a=[0,1,2]
    x = training_sample[a,:]
    #Get the Y values
    b=[3,4]
    y = training_sample[b,:]
    
    #Get the weights
    w = np.random.rand(2,3) - 0.5
    v = np.random.rand(3,2) - 0.5
    u = np.random.rand(2,3) - 0.5
    
    error = []
    for e in range(epochs):

        #Get the outputs for all layers
        
        #Get the output for the first layer
        net1 = np.dot(w,x) 
        hidden1 = np.zeros((2,100))
        hidden1 +=sigmoid(net1)

        #Get the output for the Second layer
        net2 = np.dot(v,hidden1)
        hidden2 = np.zeros((3,100))
        hidden2 +=sigmoid(net2)

        #Get the output
        output = np.zeros((2,100))
        net3 = np.dot(u,hidden2)
        output +=sigmoid(net3)


        #Backprapogation

        #output layer
        delta3 = np.zeros((2,100))
        delta3+= (y-output)*sigderivative(output)

        #second hidden layer
        delta2 = np.zeros((3,100))
        delta2 += sigderivative(hidden2)*np.dot(u.T,delta3)

        #first hidden layer
        delta1 = np.zeros((2,100))
        delta1 += sigderivative(hidden1)*np.dot(v.T,delta2)

        
        
        #Update the weights
        u+= learnrate*np.dot(delta3,hidden2.T)
        v+= learnrate*np.dot(delta2,hidden1.T)
        w+= learnrate*np.dot(delta1,x.T)
        
        e = output-y
        error.append(np.dot(e,e.T))
    return error
        


# In[8]:


training_sample = np.array([x1,x2,x3,s1,s2])
error = Backpro(training_sample,learnrate=0.1,epochs=10)

#We can see the errors is decreasing
error

