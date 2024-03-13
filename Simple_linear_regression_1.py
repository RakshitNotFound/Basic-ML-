#!/usr/bin/env python
# coding: utf-8

# ## Import our library

# In[29]:


# import the library
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
from ml_metrics import rmse
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


# loading datasets
timeData = pd.read_csv("data/delivery_time.csv")
timeData.head()


# ## Data cleaning and information

# In[31]:


# shape of the data
print('Dimenssion:', timeData.shape)


# In[32]:


# datatypes and information of the data
timeData.info()


# In[33]:


# statistical summary of the data
timeData.describe()


# In[34]:


timeData.isnull().sum()


# In[35]:


timeData[timeData.duplicated()].shape


# ## Data exploration

# In[36]:


# histogram for visualize the data distribution
plt.figure(figsize=(12,7))
plt.subplot(2, 2, 1)
timeData['Delivery Time'].hist()
plt.title('Histogram (Delivery Time)')
plt.subplot(2, 2, 2)
timeData['Sorting Time'].hist()
plt.title('Histogram (Sorting Time)')
plt.show()


# In[37]:


# check the outliers
timeData.boxplot(column=['Delivery Time'])


# In[38]:


# check the pair-wise relationships
sns.pairplot(timeData)


# In[39]:


# coorelation matrix
plt.figure(figsize = (5,4))
sns.heatmap(timeData.corr(), annot = True, cmap = 'viridis_r')
plt.title('Coorelation heatmap')


# In[40]:


# measure of coorelation
timeData.corr()


# In[41]:


# scatter plot to bvisualize the relationship between the data
timeData.plot.scatter(x = 'Delivery Time', y = 'Sorting Time')


# In[42]:


# data distribution and probability plot to check observed and expected values
plt.figure(figsize=(12,5))

plt.subplot(2, 2, 1)
timeData['Delivery Time'].hist()

plt.subplot(2, 2, 2)
stats.probplot(timeData['Delivery Time'], dist="norm", plot=plt)

plt.subplot(2, 2, 3)
timeData['Sorting Time'].hist()

plt.subplot(2, 2, 4)
stats.probplot(timeData['Sorting Time'], dist="norm", plot=plt)

plt.show()


# In[43]:


# rename the dataframes for further analysis and operations
timeData1 = timeData.rename(columns={'Sorting Time': 'sortingTime', 'Delivery Time': 'deliveryTime'})
timeData1.head()


# In[44]:


# to check the heteroscedasticity of residuals (fitted value against residuals)
sns.residplot(x = 'sortingTime', y = 'deliveryTime', data = timeData1, lowess = True)


# ## Build a model

# In[45]:


# model1 and summary (Transformation: normal)
model1 = smf.ols("deliveryTime ~ sortingTime", data = timeData1).fit()
model1.summary()


# In[46]:


# model1 predicted data
predict1 = model1.predict(timeData1.sortingTime)
predict1.head()


# In[47]:


# calculate prediction error (RMSE)
rmseValue1 = rmse(predict1, timeData1.deliveryTime) 
print(rmseValue1)


# In[48]:


model1.params


# In[49]:


# model2 and summary (Transformation: logarithamic)
model2=smf.ols("deliveryTime ~ np.log(sortingTime) + 1", data = timeData1).fit()
model2.summary()


# In[50]:


# model2 predicted data
predict2 = model2.predict(timeData1.sortingTime)
predict2.head()


# In[51]:


# RMSE (Root Mean Square Error)
rmseValue2 = rmse(predict2, timeData1.deliveryTime) 
print(rmseValue2)


# In[52]:


model2.params


# In[53]:


# model3 and summary (Transformation: Square root)
def sRT(x):
    return x**(1/2)

model3 =smf.ols("deliveryTime ~ sRT(sortingTime) + 1", data = timeData1).fit()
model3.summary()


# In[54]:


# model3 predicted values
predict3 = model3.predict(timeData1.sortingTime)
print('Predicted delivery time:\n')
predict3


# In[55]:


# RMSE error value
rmseValue3 = rmse(predict3, timeData1.deliveryTime) 
print(rmseValue3)


# ## Model summary and selection

# In[56]:


# Regression line is drawn using predicted values for different models
plt.figure(figsize=(12,7))

plt.subplot(2, 2, 1)
plt.scatter(x = timeData1.sortingTime, y = timeData1.deliveryTime, color='blue')
plt.plot(timeData1.sortingTime, predict1, color='black')
plt.xlabel("Delivery time")
plt.ylabel("Sorting time")
plt.title('Model1')

plt.subplot(2, 2, 2)
plt.scatter(x = timeData1.sortingTime, y = timeData1.deliveryTime, color='blue')
plt.plot(timeData1.sortingTime, predict2, color='black')
plt.xlabel("Delivery time")
plt.ylabel("Sorting time")
plt.title('Model2')

plt.subplot(2, 2, 3)
plt.scatter(x = timeData1.sortingTime, y = timeData1.deliveryTime, color='blue')
plt.plot(timeData1.sortingTime, predict3, color='black')
plt.xlabel("Delivery time")
plt.ylabel("Sorting time")
plt.title('Model3')

plt.show()


#       [Models]   |    [R^2]    |  p-value  |    [RMSE]   |   [Transformation type]
#     ----------------------------------------------------------------------------------
#     1) model1     0.682         0.001       2.7916         withput transformation
#     2) model2     0.695         0.642       2.7331         logarithamic transformation
#     3) model3     0.696         0.411       2.7315         square root transformation

# ###### Best fit model is 'model3' with accuracy of 69.60% and error measures of 2.7315
# * model accuracy: 69.60%
# * error prediction: 2.7315
# 
# ###### Predicted delivery time (based on model3)
# * 22.578867
# * 13.354345
# * 16.921761
# * 21.290936
# * 22.578867
