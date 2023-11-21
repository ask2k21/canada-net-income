#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('canada_per_capita_income.csv')
df.head()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Year')
plt.ylabel('Per_capita_income')
plt.scatter(df.year,df.pci,color='red',marker='+')


# In[8]:


df.columns


# In[9]:


df.rename(columns = {'per capita income (US$)':'pci'},inplace = True)


# In[10]:


df.columns


# In[15]:


X = df.drop('pci',axis='columns')
y = df.pci


# In[16]:


reg = linear_model.LinearRegression()
reg.fit(X,y)


# In[17]:


reg.predict([[2023]])


# In[ ]:




