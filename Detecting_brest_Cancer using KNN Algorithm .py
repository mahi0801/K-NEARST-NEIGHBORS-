#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()


# In[3]:


cancer


# In[4]:


cancer["frame"]


# In[5]:


cancer["data"]


# In[6]:


cancer["target"]


# In[7]:


cancer["feature_names"]


# In[8]:


cancer.keys()


# In[9]:


cancer.values()


# In[10]:


print(cancer['DESCR'])


# In[11]:


df =pd.DataFrame(np.c_[cancer["data"],cancer["target"]],columns=np.append(cancer["feature_names"],["target"]))
df


# In[12]:


df.isna().sum()


# In[13]:


sns.countplot(df["target"])


# In[14]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[15]:


x


# In[16]:


y


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[19]:


x_train


# In[20]:


y_train


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =25 )
classifier.fit(x_train,y_train)


# In[22]:


y_pred = classifier.predict(x_test)


# In[23]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


# In[24]:


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac


# In[25]:


sns.heatmap(cm,annot=True)


# In[ ]:




