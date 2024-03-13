#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
 Calculate Mean, Median, Mode, Variance, Standard Deviation, Range
    Comment about the values / draw inferences, for the given dataset Points, Score, Weigh
    Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values / Draw some inferences.
'''


# In[ ]:


# importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# reading the dataset
df1 = pd.read_csv ('Q7.csv')
df1


# In[ ]:


# show the information of data
df1.info()


# In[ ]:


# show the shape of data
print('Shape :', df1.shape)


# In[ ]:


# Check the null values
df1.isnull().head()


# In[ ]:


# count the values and sum
print('1) sumPoint : ',df1['Points'].sum(),'\n2) sumScore : ',df1['Score'].sum(),',\n3) sumWeight: ',df1['Weigh'].sum())
print('\n1) sumPoint :',df1['Points'].count(),'\n2) sumScore :',df1['Score'].count(),'\n3) sumWeight:',df1['Weigh'].count())


# In[ ]:


# Calculate mean values
meanPoint = df1['Points'].mean()
meanScore = df1['Score'].mean()
meanWeight = df1['Weigh'].mean()

print('Mean of Point DataFrame  :', meanPoint)
print('Mean of Score DataFrame  :', meanScore)
print('Mean of Weight DataFrame :', meanWeight)


# In[ ]:


# Calculate median values
medianPoint = df1['Points'].median()
medianScore = df1['Score'].median()
medianWeight = df1['Weigh'].median()

print('Median of Point DataFrame  :', medianPoint)
print('Median of Score DataFrame  :', medianScore)
print('Median of Weight DataFrame :', medianWeight)


# In[ ]:


# Calculate mod values
modPoint = df1['Points'].mode()
modScore = df1['Score'].mode()
modWeight = df1['Weigh'].mode()

print('Mode of Point DataFrame -->\n', modPoint)
print('\nMode of Score DataFrame -->\n',modScore)
print('\nMode of Weight DataFrame ->\n', modWeight)


# In[ ]:


# Calculate variances
varPoint = df1['Points'].var()
varScore = df1['Score'].var()
varWeight = df1['Weigh'].var()

print('Variance of Point DataFrame :', varPoint)
print('Variance of Score DataFrame :', varScore)
print('Variance of Weight DataFrame :', varWeight)


# In[ ]:


# Calculate standard variences
stdPoint = df1['Points'].std()
stdScore = df1['Score'].std()
stdWeight = df1['Weigh'].std()

print('Std. Variance of Point DataFrame  :', stdPoint)
print('Std. Variance of Score DataFrame  :', stdScore)
print('Std. Variance of Weight DataFrame :', stdWeight)


# In[ ]:


# Calculate range
maxPoint = df1['Points'].max()
minPoint = df1['Points'].min()
rangePoint = maxPoint - minPoint
print('Range of Point DataFrame  :', rangePoint)

maxScore = df1['Score'].max()
minScore = df1['Score'].min()
rangeScore = maxScore - minScore
print('Range of Score DataFrame  :', rangeScore)

maxWeight = df1['Weigh'].max()
minWeight = df1['Weigh'].min()
rangeWeight = maxWeight - minWeight
print('Range of Weight DataFrame :', rangeWeight)


# In[ ]:


# Visualization first 20 values
df1['10_Points'] = df1['Points'].rolling(window = 20).mean()
df1['10_Score'] = df1['Score'].rolling(window = 20).mean()


# In[ ]:


# plotting the graph 
df1[['Points', 'Score']].plot(figsize = (10,5))


# In[ ]:


'''
Q8) Calculate Expected Value for the problem below
    a) The weights (X) of patients at a clinic (in pounds)
       108, 110, 123, 134, 135, 145, 167, 187, 199
    Assume one of the patients is chosen at random. What is the Expected Value of the Weight of that patient?
'''


# In[ ]:


import statistics 
wightsOfPatients = (108, 110, 123, 134, 135, 145, 167, 187, 199)


# In[ ]:


expectedValue = statistics.mean(wightsOfPatients)
print('Expected value of the patient :', expectedValue)


# In[ ]:


'''
Q9) Calculate Skewness, Kurtosis & draw inferences on the following data
    Cars speed and distance --> Use Q9_a.csv
    SP and Weight(WT) --> Use Q9_b.csv
'''


# In[ ]:


# import library for visualizations
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# read datasets
df2 = pd.read_csv ('Q9_a.csv')
df3 = pd.read_csv('Q9_b.csv')


# In[ ]:


df2.columns = ['index', 'speed', 'dist']
df2.head()


# In[ ]:


df3


# In[ ]:


df2.info()
df3.info()


# In[ ]:


# check the shape of data
print('Shape :', df2.shape)
print('Shape :', df3.shape)


# In[ ]:


# check the null values
df2.isnull().sum()


# In[ ]:


# check the null values
df3.isnull().sum()


# In[ ]:


sn.barplot(x ='speed', y = 'dist', data = df2) 


# In[ ]:


# Calculate Skewness and Kurtosis 
dataFrame = pd.DataFrame(data = df2);
skewValue = dataFrame.skew(axis = 0)  # OR df2['speed'].skew()  
kurtValue = dataFrame.kurt(axis = 0)  # OR df2['speed'].kurt()
print('Skewness values :', skewValue)
print('\nKurtosis values :', kurtValue)


# In[ ]:


# Visualization (matplotlib)
plt.subplot(1,2,1)
plt.hist(df2['speed'])
plt.xlabel('speed')

plt.subplot(1,2,2)
plt.hist(df2['dist'])
plt.xlabel('dist')


# In[ ]:


# visualization (seaborn)
plt.subplot(1,2,1)
sn.distplot(df2['speed'])

plt.subplot(1,2,2)
sn.distplot(df2['dist'])


# In[ ]:


sn.barplot(x ='SP', y = 'WT', data = df3) 


# In[ ]:


# Calculate Skewness and Kurtosis 
dataFrame = pd.DataFrame(data = df3);
skewValue = dataFrame.skew(axis = 0)  # OR df2['SP'].skew()  
kurtValue = dataFrame.kurt(axis = 0)  # OR df2['SP'].kurt()
print('Skewness values :', skewValue)
print('\nKurtosis values :', kurtValue)


# In[ ]:


# graph plot
plt.subplot(1,2,1)
plt.hist(df3['SP'])
plt.xlabel('SP')

plt.subplot(1,2,2)
plt.hist(df3['WT'])
plt.xlabel('WT')


# In[ ]:


# graph plot
plt.figure(figsize = (8,6))
sn.distplot(df3['SP'])

plt.figure(figsize = (8,6))
sn.distplot(df3['WT'])


# In[ ]:


'''
Q11) Suppose we want to estimate the average weight of an adult male in Mexico. We draw a random 
     sample of 2,000 men from a population of 3,000,000 men and weigh them. We find that the 
     average person in our sample weighs 200 pounds, and the standard deviation of the sample is 
     30 pounds. Calculate 94%,98%,96% confidence interval?
'''


# In[ ]:


import math
from scipy import stats

# given data
X_bar = 200 # sample mean
n = 2000 # random samples
S = 3000000 # population
sigma = 30 # std deviation
# confidenceInterval_94, confidenceInterval_98, confidenceInterval_96 = ?, ?, ?

# Formulae: 
# confidenceInterval = X_bar +- (Z (1- alpha) * (sigma / sqrt(n)))
# Zvalue = stats.norm.ppf()  # (1- alpha)

sigmaByRootN = (sigma / math.sqrt(2000))
print(sigmaByRootN)


# In[ ]:


# confidenceInterval_94
# 94 -->   3 + 94 + 3 = 100
# we are interested at 97 value which is count from 0 to 97

Zvalue_97 = stats.norm.ppf(0.97)
print('Z value =', Zvalue_97)

confidenceInterval_94 = Zvalue_97 * sigmaByRootN
# print(confidenceInterval_94)
print('Confidence Interval at 94 % --> Upper bound :', X_bar + confidenceInterval_94, 'Lower bound :',  X_bar - confidenceInterval_94)


# In[ ]:


Zvalue_98 = stats.norm.ppf(0.98)
print('Z value =', Zvalue_98)

confidenceInterval_96 = Zvalue_98 * sigmaByRootN
# print(confidenceInterval_96)
print('Confidence Interval at 96 % --> Upper bound :', X_bar + confidenceInterval_96, 'Lower bound :',  X_bar - confidenceInterval_96)


# In[ ]:


Zvalue_99 = stats.norm.ppf(0.99)
print('Z value =', Zvalue_99)

confidenceInterval_98 = Zvalue_99 * sigmaByRootN
# print(confidenceInterval_98)
print('Confidence Interval at 98 % --> Upper bound :', X_bar + confidenceInterval_98, 'Lower bound :',  X_bar - confidenceInterval_98)


# In[ ]:


'''
Q12) Below are the scores obtained by a student in tests 
     34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
     1) Find mean, median, variance, standard deviation.
     2) What can we say about the student marks? 
'''


# In[ ]:


import statistics


# In[ ]:


# Statistical analysis and calculations
scores = (34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56)
print('Mean      : ', statistics.mean(scores))
print('Median    : ', statistics.median(scores))
print('Mode      : ', statistics.mode(scores))
print('Multimode : ', statistics.multimode(scores))
print('Mode      : ', statistics.variance(scores))
print('Mode      : ', statistics.stdev(scores))
print('Min value : ', min(scores))
print('Max value : ', max(scores))
print('Difference: ', max(scores) - min(scores))


# In[ ]:


sn.distplot(scores)
plt.xlabel('Student Scores')


# In[ ]:


plt.figure(figsize = (6,4))
plt.hist(scores)
plt.xlabel('Student Scores')


# In[ ]:


# comment: from above visualization, here we notice that the distribution data is positively skewed
#          As the tail of the distribution is at right side


# In[ ]:


'''
Q 20) Calculate probability from the given dataset for the below cases (Data _set: Cars.csv)
      Calculate the probability of MPG  of Cars for the below cases.
      MPG <- Cars$MPG
      a) P (MPG>38)
      b) P (MPG<40)
      c) P (20<MPG<50)
'''


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df4 = pd.read_csv ('Cars.csv')
df4.head()


# In[ ]:


print('Shape :',df4.shape)
df4.isnull().sum()


# In[ ]:


meanMu = df4['MPG'].mean()
stdDeviation = df4['MPG'].std()
print('mu    :', meanMu, '\nsigma :',stdDeviation)


# In[ ]:


# P(MPG > 38)
Z1 = (38 - meanMu) / stdDeviation  # Formula: Z = (X - mu) / sigma
Zvalue1 = stats.norm.cdf(Z1)  # Zvalue1 = Z / (1 - Alpha)
print('Probalibity (MPG > 38)  :', (1 - Zvalue1) * 100 ,'%')

# P(MPG < 40)
Z2 = (40 - meanMu) / stdDeviation  
Zvalue2 = stats.norm.cdf(Z2)  
print('Probalibity (MPG < 40)  :', (Zvalue2 * 100) ,'%')

# P(20 < MPG < 50) --> Using P(X1 < MPG) - P(X2 < MPG)
Z3 = (20 - meanMu) / stdDeviation  
Zvalue3 = stats.norm.cdf(Z3)
Z4 = (50 - meanMu) / stdDeviation  
Zvalue4 = stats.norm.cdf(Z4)
print('Probalibity (20<MPG<50) :', ((Zvalue4 - Zvalue3) * 100) ,'%')


# In[ ]:


'''
Q 21) Check whether the data follows normal distribution
      a) Check whether the MPG of Cars follows Normal Distribution 
         Dataset: Cars.csv
      b) Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist) from wc-at data set follows Normal Distribution 
         Dataset: wc-at.csv
'''


# In[ ]:


df5 = pd.read_csv ('Cars.csv')
df4.head()


# In[ ]:


df6 = pd.read_csv('wc-at.csv')
df6.head()


# In[ ]:


df5.describe()


# In[ ]:


print('MPG meadian:', df5['MPG'].median())


# In[ ]:


print('MPG mode :',df5['MPG'].mode())


# In[ ]:


sn.distplot(df5.MPG, label = 'MPG')
plt.legend()

# comment : MPG car data does not follow the exactly normal distribution


# In[ ]:


sn.distplot(df6.AT, label = 'AT')
sn.distplot(df6.Waist, label = 'Waist')
plt.xlabel('AT')
plt.ylabel('Waist')
plt.legend()

# comment: AT data and Waist data does not follow the exactly normal distribution


# In[ ]:


'''
Q22 Calculate the Z scores of 90% confidence interval, 94% confidence interval and 60% confidence interval 
'''


# In[ ]:


from scipy import stats

# Z scores of 90% confidence interval
print('Z score at 90% CI :', stats.norm.ppf(0.95)) # summary 5 <-- 90 --> 5

# Z scores of 94% confidence interval
print('Z score at 94% CI :', stats.norm.ppf(0.97)) # summary 3 <-- 94 --> 3 

# Z scores of 60% confidence interval
print('Z score at 60% CI :', stats.norm.ppf(0.80)) # summary 20 <-- 60 --> 20


# In[ ]:


'''
Q23 Calculate the t scores of 95% confidence interval, 96% confidence interval and 99% confidence interval 
    for the sample size of 25
'''


# In[ ]:


from scipy import stats

# n = 25
# C.I = (1 - Alpha)

# t scores of 95% confidence interval 
print('t scores of 95% confidence interval :', stats.t.ppf(0.975, df = 24))  # 2.5 <-- 95 --> 2.5 
# t scores of 96% confidence interval 
print('t scores of 96% confidence interval :', stats.t.ppf(0.98, df = 24))  # 2 <-- 96 --> 2 
# t scores of 99% confidence interval 
print('t scores of 99% confidence interval :', stats.t.ppf(0.995, df = 24))  # 0.5 <-- 99 --> 0.5 


# In[ ]:


'''
Q 24) A Government company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days
      Hint:  
            rcode -> pt(tscore, df)  
            df -> degrees of freedom
'''


# In[ ]:


# n = 18, sigma = 270, s (std. Dev) = 90, X (bar) = 260, Degree of freedom = 17

print('Value:', stats.t.cdf(0.975, df = 17))


# In[ ]:


# End of assignment 1

