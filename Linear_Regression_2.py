#!/usr/bin/env python
# coding: utf-8

# ## Import our library

# In[42]:


# import the library
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.formula.api as smf
from ml_metrics import rmse
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


# loading datasets
salaryData = pd.read_csv("data/Salary_Data.csv")
salaryData.head()


# ## Data cleaning and information

# In[44]:


# shape of the data
print('Dimenssion:', salaryData.shape)


# In[45]:


# datatypes and information of the data
salaryData.info()


# In[46]:


# statistical summary of the data
salaryData.describe()


# In[47]:


salaryData.isnull().sum()


# In[48]:


salaryData[salaryData.duplicated()].shape


# ## Data exploration

# In[49]:


# histogram for visualize the data distribution
plt.figure(figsize=(12,7))
plt.subplot(2, 2, 1)
salaryData['YearsExperience'].hist()
plt.title('Histogram (YearsExperience)')
plt.subplot(2, 2, 2)
salaryData['Salary'].hist()
plt.title('Histogram (Salary)')
plt.show()


# In[50]:


# check the outliers
plt.figure(figsize=(12,8))

plt.subplot(2, 2, 1)
salaryData.boxplot(column=['YearsExperience'])

plt.subplot(2, 2, 2)
salaryData.boxplot(column=['Salary'])

plt.show()


# In[51]:


# check the pair-wise relationships
sns.pairplot(salaryData)


# In[52]:


# coorelation matrix
plt.figure(figsize = (5,4))
sns.heatmap(salaryData.corr(), annot = True)
plt.title('Coorelation heatmap', color='blue')


# In[53]:


# measure of coorelation
salaryData.corr()


# In[54]:


# scatter plot to bvisualize the relationship between the data
salaryData.plot.scatter(x = 'YearsExperience', y = 'Salary')


# In[55]:


# to check the residuals (fitted value against residuals) : heteroscedasticity
sns.residplot(x = 'YearsExperience', y = 'Salary', data = salaryData, lowess = True)


# * Residual plot shows fairly random pattern
# * No U-shaped pattern or O-shaped pattern is found 
# * This random pattern indicates that a linear model provides a decent fit to the data.

# ## Build a model

# In[56]:


# model1 and summary (Transformation: normal)
model1 = smf.ols("Salary ~ YearsExperience", data = salaryData).fit()
model1.summary()


# In[57]:


# model1 predicted and RMSE value
predict1 = model1.predict(salaryData.YearsExperience)
rmseValue1 = rmse(predict1, salaryData.Salary) 
print(rmseValue1)


# In[58]:


model1.params


# In[59]:


# regression plots for model1
import statsmodels.api as sm

fig = plt.figure(figsize =(12,7))
fig = sm.graphics.plot_regress_exog(model1, 'YearsExperience', fig = fig)


# In[60]:


# model2 and summary (Transformation: logarithamic)
model2=smf.ols("Salary ~ np.log(YearsExperience) + 1", data = salaryData).fit()
model2.summary()


# In[61]:


# model2 predicted and RMSE value
predict2 = model2.predict(salaryData.YearsExperience)
rmseValue2 = rmse(predict2, salaryData.Salary) 
print(rmseValue2)


# In[62]:


model2.params


# In[63]:


# regression plot for model2
fig = plt.figure(figsize =(12,7))
fig = sm.graphics.plot_regress_exog(model2, 'np.log(YearsExperience)', fig = fig)


# In[64]:


# model3 and summary (Transformation: reciprocal)
def rT(x):
    return 1/x

model3 =smf.ols("Salary ~ rT(YearsExperience)", data = salaryData).fit()
model3.summary()


# In[65]:


# model3 predicted and RMSE value
predict3 = model3.predict(salaryData.YearsExperience)
rmseValue3 = rmse(predict3, salaryData.Salary) 
print(rmseValue3)


# In[66]:


model3.params


# In[67]:


# regression plot for model3
fig = plt.figure(figsize =(12,7))
fig = sm.graphics.plot_regress_exog(model3, 'rT(YearsExperience)', fig = fig)


# In[68]:


# model4 and summary (Transformation: Square root)
def sRT(x):
    return x**(1/2)

model4 =smf.ols("Salary ~ sRT(YearsExperience)", data = salaryData).fit()
model4.summary()


# In[69]:


# model4 predicted and RMSE value
predict4 = model4.predict(salaryData.YearsExperience)
rmseValue4 = rmse(predict4, salaryData.Salary) 
print(rmseValue4)


# In[70]:


model4.params


# In[71]:


# regression plot for model4
fig = plt.figure(figsize =(12,7))
fig = sm.graphics.plot_regress_exog(model4, 'sRT(YearsExperience)', fig = fig)


# In[72]:


# model5 and summary (Transformation: exponential)
def eT(x):
    return x**(1/5)

model5=smf.ols("Salary ~ eT(YearsExperience)", data = salaryData).fit()
model5.summary()


# In[73]:


# model5 predicted and RMSE value
predict5 = model5.predict(salaryData.YearsExperience)
rmseValue5 = rmse(predict5, salaryData.Salary) 
print(rmseValue5)


# In[74]:


model5.params


# In[75]:


# regression plot for model5
fig = plt.figure(figsize =(12,7))
fig = sm.graphics.plot_regress_exog(model5, 'eT(YearsExperience)', fig = fig)


# ## Model summary and selection

# In[76]:


# Regression line is drawn using predicted values for different models
plt.figure(figsize=(12,7))

plt.subplot(3, 2, 1)
plt.scatter(x = salaryData.YearsExperience, y = salaryData.Salary, color='blue')
plt.plot(salaryData.YearsExperience, predict1, color='black')
plt.xlabel("Year exp")
plt.ylabel("Salary")
plt.title('Model1')

plt.subplot(3, 2, 2)
plt.scatter(x = salaryData.YearsExperience, y = salaryData.Salary, color='blue')
plt.plot(salaryData.YearsExperience, predict2, color='black')
plt.xlabel("Year exp")
plt.ylabel("Salary")
plt.title('Model2')

plt.subplot(3, 2, 3)
plt.scatter(x = salaryData.YearsExperience, y = salaryData.Salary, color='blue')
plt.plot(salaryData.YearsExperience, predict3, color='black')
plt.xlabel("Year exp")
plt.ylabel("Salary")
plt.title('Model3')

plt.subplot(3, 2, 4)
plt.scatter(x = salaryData.YearsExperience, y = salaryData.Salary, color='blue')
plt.plot(salaryData.YearsExperience, predict4, color='black')
plt.xlabel("Year exp")
plt.ylabel("Salary")
plt.title('Model4')

plt.subplot(3, 2, 5)
plt.scatter(x = salaryData.YearsExperience, y = salaryData.Salary, color='blue')
plt.plot(salaryData.YearsExperience, predict5, color='black')
plt.xlabel("Year exp")
plt.ylabel("Salary")
plt.title('Model5')

plt.show()


#       [Models]   |    [R^2]    |  p-value  |    [RMSE]   |   [Transformation type]
#     ----------------------------------------------------------------------------------
#     1) model1     0.957         0.000       5592.04       without transformation
#     2) model2     0.854         0.007       10302.89      logarithamic transformation
#     3) model3     0.589         0.000       17288.30      reciprocal transformation
#     4) model4     0.931         0.003       7080.09       squreroot transformation
#     5) model5     0.891         0.000       8090.77       exponential transformation

# ###### Best fit model is 'model1' with accuracy of 95.70% and error measures of 5592.04
# * model accuracy: 95.70%
# * error prediction: 5592.04
# 
# ###### Predicted salary (based on model1)
# * 36187.158752
# * 38077.151217
# * 39967.143681
# * 44692.124842
# * 46582.117306
