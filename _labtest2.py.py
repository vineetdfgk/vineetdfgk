#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import sklearn.datasets as dt

data=dt.load_iris().data
df=pd.DataFrame(data)

print(df.head(10))




# In[13]:


summary = df.describe().loc[['instances','feature','target_classes']]
summary
num_instances=iris_df.shapes[0]
num_feature=iris_df.shapes[1]-1
num_classes=len(iris.target_names)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size(0.2))


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(correlation_matrix)
plt.show()


# In[ ]:


k_values =range(1,21)
cv_score=[]

for k in k_values:
    km=

