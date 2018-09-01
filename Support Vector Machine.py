
# coding: utf-8
# Creater: Jinwei Wang
# Revised Date: 9-1-2018
# In[1]:


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


#load training data
q1xfile = '02_τ¼¼Σ╕ëτ½á_q1x.dat'
q1yfile = '02_τ¼¼Σ╕ëτ½á_q1y.dat'

q1x = np.loadtxt(q1xfile,dtype = float)
q1y = np.loadtxt(q1yfile,dtype = float)


# In[3]:


#define SVM Parameters
x = q1x
y = 2*(q1y-0.5)
C = cp.Parameter()
C.value = 1
m = q1x.shape[0]


# In[4]:


#Plot the data
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='plasma')
plt.show()


# In[5]:


#define SVM variables
w = cp.Variable(m,1)
xi = cp.Variable(m,2)
b = cp.Variable()

print(x.shape)
print(y.shape)


# In[6]:


#train SVM Model using cvxpy
objective = cp.Minimize(0.5*cp.norm(w,2) + C*cp.sum_entries(xi))
xicons = [xi >= 0 ]
ycons = [y*(w.T*x + b) >= 1 - xi ]
constraints = xicons + ycons
prob = cp.Problem(objective, constraints)
prob.solve()
print("Problem Status: %s"%prob.status)


# In[7]:


print(prob.value)

