#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


url= "https://raw.githubusercontent.com/DrSaadLa/PythonTuts/main/Data/auto.csv"


# In[4]:


data=pd.read_csv(url)


# In[9]:


data.shape


# In[7]:


data.info()


# In[12]:


df=pd.DataFrame(data)
df


# In[33]:


type(df)
print(dir(df))
print(df.keys)
print(df.shape)
df.info()
df.describe()


# In[23]:


df.describe().T


# In[24]:


df.head(5)
df.tail(5)


# In[31]:


df.plot()
plt.show()


# In[32]:


sns.heatmap(df.corr(), square=True, cmap='RdYlGn',center=True,annot=True)


# In[37]:


df.corr()


# In[44]:


## Import Ridge
from sklearn.linear_model import Ridge
## Import the GridSearchCV
from sklearn.model_selection import GridSearchCV
## Instantiate the estimator object
r_reg = Ridge(df)
r_reg


# In[48]:


# Instantiate a ridge object
r_obj = Ridge(alpha = range(0,1),fit_intercept = True,normalize = True)


# In[ ]:




