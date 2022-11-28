#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

cancer

cancer["frame"]
cancer["data"]
cancer["target"]


cancer["feature_names"]

cancer.keys()

cancer.values()

print(cancer['DESCR'])

df =pd.DataFrame(np.c_[cancer["data"],cancer["target"]],columns=np.append(cancer["feature_names"],["target"]))
df

df.isna().sum()

sns.countplot(df["target"])


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

x
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


x_train

y_train

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =25 )
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

sns.heatmap(cm,annot=True)




