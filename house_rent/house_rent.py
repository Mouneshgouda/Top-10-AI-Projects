#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


# In[2]:


dataset = pd.read_csv('house_rent.csv')


# In[3]:


X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]


# In[6]:


regressor = LinearRegression()


# In[7]:


regressor.fit(X, y)


# In[8]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[9]:


model = pickle.load(open('model.pkl','rb'))


# In[10]:


print(model.predict([[1, 1, 2]]))


# In[ ]:




