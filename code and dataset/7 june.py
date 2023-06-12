#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('whitegrid')


# In[2]:


dataset = pd.read_csv('dummy.csv')


# In[3]:


dataset


# In[4]:


dataset['Attendance'] = dataset['Attendance'].map({'yes':1'No':0})


# In[5]:


dataset['Attendance'] = dataset['Attendance'].map({'Yes':1,'No':0})


# In[6]:


dataset


# In[7]:


x = dataset[['SAT','Attendance']]


# In[8]:


x


# In[9]:


y = dataset['GPA']


# In[10]:


y


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


model = LinearRegression()


# In[13]:


model.fit(x,y)


# In[14]:


test = pd.DataFrame([[1600,1],[1600,0]],columns=['SAT','Attendance'])


# In[15]:


test


# In[16]:


model.predict(test)


# In[17]:


#bo
model.intercept_


# In[18]:


#b1,b2
model.coef_


# In[19]:


plt.scatter(dataset['SAT'],dataset['GPA'])
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# In[20]:


y_hat_yes = 0.6438+0.0013*dataset['SAT']+ 0.2226*1
y_hat_no = 0.6438+0.0013*dataset['SAT']+ 0.2226*0


# In[21]:


plt.scatter(dataset['SAT'],dataset['GPA'])
plt.plot(dataset['SAT'],y_hat_yes,color='green')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# In[22]:


plt.scatter(dataset['SAT'],dataset['GPA'])
plt.plot(dataset['SAT'],y_hat_no,color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# In[23]:


plt.scatter(dataset['SAT'],dataset['GPA'])
plt.plot(dataset['SAT'],y_hat_yes,color='green')
plt.plot(dataset['SAT'],y_hat_no,color='red')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




