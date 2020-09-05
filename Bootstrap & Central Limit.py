#!/usr/bin/env python
# coding: utf-8

# # 1. The Bootstrap and The Normal Curve
# 
# In this exercise, we will explore a dataset that includes the safety inspection scores for restaurants in the city of Austin, Texas. We will be interested in determining the average restaurant score for the city from a random sample of the scores; the average restaurant score is out of 100. We'll compare two methods for computing a confidence interval for that quantity: the bootstrap resampling method, and an approximation based on the Central Limit Theorem.

# In[1]:


import numpy as np
from datascience import *

import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import warnings
warnings.simplefilter('ignore', FutureWarning)


# In[2]:


pop_restaurants = Table.read_table("/Users/shristidhakal/Documents/Grad School/INFO5502/Restaurants Ratings/restaurant_inspection_scores.csv").drop('Facility ID','Process Description')
pop_restaurants


# In[3]:


pop_restaurants.hist('Score')


# In[4]:


pop_mean = np.mean(pop_restaurants.column('Score'))
pop_mean


# In[5]:


restaurant_sample = pop_restaurants.sample(100, with_replacement=False)
restaurant_sample


# In[6]:


restaurant_sample.hist('Score')


# In[7]:


sample_mean = np.mean(restaurant_sample.column('Score'))
sample_mean


# # 1.1 Question 1
# 
# Complete the function one_resampled_mean below. It should take in an original table data, with a column Score, and return the mean score of one resampling from data.

# In[8]:


def one_resampled_mean(data):
    resampled_data = data.sample(100, with_replacement=False)
    sample_mean = np.mean(resampled_data.column('Score'))
    return sample_mean

this_mean = one_resampled_mean(pop_restaurants)
this_mean


# # 1.2 Question 2
# 
# Complete the function bootstrap_scores below. It should take no arguments. It should simulate drawing 5000 resamples from restaurant_sample and compute the mean restaurant score in each resample. It should return an array of those 5000 resample means.

# In[9]:


def bootstrap_scores():
    resampled_means = make_array()
    for i in range(5000):
        resampled_data = restaurant_sample.sample(100, with_replacement=True)
        resampled_mean = one_resampled_mean(resampled_data)
        resampled_means = np.append(resampled_means, resampled_mean)
    return resampled_means

resampled_means = bootstrap_scores()
resampled_means


# In[10]:


Table().with_column('Resampled Means', resampled_means).hist()


# # 1.3 Question 3
# 
# Compute a 95 percent confidence interval for the average restaurant score using the array resampled_means.

# In[11]:


lower_bound = np.round(percentile(2.5, resampled_means), 2)
upper_bound = np.round(percentile(97.5, resampled_means), 2)
print("95% confidence interval for the average restaurant score, computed by bootstrapping",
      ":\n(",lower_bound, ",", upper_bound, ")")


# # 1.4  Question 4
# 
# What distribution is the histogram between question 2 and 3 displaying (that is, what data are plotted), and why does it have that shape?

# >The histogram is displaying the means of 5000 bootstrapped resamples drawn with replacement from our original restaurant samples of 100, which is also drawn with replacement. It is roughly normally distributed because of the Central Limit Theorem i.e. if we repeatedly calculate the arithmetic average of samples drawn randomly from a population with replacement, the probability distribution of those averages tends to follow a normal distribution even if the population's distribution is not normal.     

# # 1.5  Question 5
# Does the distribution of the sampled scores look normally distributed?  State “yes” or “no” and describe in one sentence why you should expect this result.

# >No, the distribution of the sampled scores does not look normally distributed, instead it is left-tailed with most observations centered towards the right (close to the value of 100). This is because the distribution of samples does not necessarily equate to the distribution of mean of the samples, hence in this case, the Central Limit Theorem does not apply.  

# # 1.6  Question 6
# Without referencing the array resampled_means or performing any new simulations, calculate an interval around the sample_mean that covers approximately 95% of the numbers in the resampled_means array.  This confidence interval should look very similar to the one you computed in Question 3.

# In[12]:


sample_sd = np.std(restaurant_sample.column('Score'))
sample_sd


# In[13]:


sample_mean = np.mean(restaurant_sample.column('Score'))
sample_sd = np.std(restaurant_sample.column('Score'))
sample_size = restaurant_sample.num_rows

sd_of_means = sample_sd * 2
lower_bound_normal = sample_mean - sd_of_means
upper_bound_normal = sample_mean + sd_of_means
print("95% confidence interval for the average restaurant score, computed by a normal approximation",
      ":\n(",lower_bound_normal, ",", upper_bound_normal, ")")


# # 2  Testing the Central Limit Theorem
# To recap the properties we just saw: The Central Limit Theorem tells us that the probability distribution of the sum or average of a large random sample drawn with replacement will be roughly normal, regardless of the distribution of the population from which the sample is drawn.

# # 2.1  Question 1
# Define the function one_statistic_prop_heads which should return exactly one simulated statistic of the proportion of heads from n coin flips.

# In[14]:


coin_proportions = make_array(.5, .5)

def one_statistic_prop_heads(n):
    simulated_proportions = sample_proportions(n, coin_proportions)
    prop_heads = simulated_proportions.item(0)* 100
    return prop_heads 

one_statistic_prop_heads(100)


# # 2.2 Question 2 
# The CLT only applies when sample sizes are "sufficiently large." This isn't a very precise statement. Is 10 large? How about 50? The truth is that it depends both on the original population distribution and just how "normal" you want the result to look. Let's use a simulation to get a feel for how the distribution of the sample mean changes as sample size goes up.
# 
# Consider a coin flip. If we say Heads is $1$ and Tails is $0$, then there's a 50% chance of getting a 1 and a 50% chance of getting a 0, which definitely doesn't match our definition of a normal distribution. The average of several coin tosses, where Heads is 1 and Tails is 0, is equal to the proportion of heads in those coin tosses (which is equivalent to the mean value of the coin tosses), so the CLT should hold true if we compute the sample proportion of heads many times.
# 
# Write a function called sample_size_n that takes in a sample size $n$. It should return an array that contains 5000 sample proportions of heads, each from $n$ coin flips.

# In[15]:


def sample_size_n(n):
    coin_proportions = make_array(.5, .5)
    heads_proportions = make_array()
    for i in np.arange(5000):
        simulated_proportions = sample_proportions(n, coin_proportions)
        prop_heads = simulated_proportions.item(0)* 100
        heads_proportions = np.append(heads_proportions, prop_heads)
    return heads_proportions

sample_size_n(5000)


# # 2.3  Question 3
# Write a function called empirical_sample_mean_sd that takes a sample size n as its argument.  The function should simulate 500 samples with replacement of size n from the flight delays dataset, and it should return the standard deviation of the means of those 500 samples.

# In[16]:


delays = Table().read_table("/Users/shristidhakal/Documents/Grad School/INFO5502/Restaurants Ratings/united_summer2015.csv")
print(delays)


# In[17]:


def empirical_sample_mean_sd(n):
    sample_means = make_array()
    
    for i in np.arange(500):
        sample = delays.select('Delay').sample(n, with_replacement=False).column(0)
        sample_mean = np.mean(sample)
        sample_means = np.append(sample_means, sample_mean)
    return np.std(sample_means)

empirical_sample_mean_sd(10)


# # 2.4 Question 4 
# Now, write a function called predict_sample_mean_sd to find the predicted value of the standard deviation of means according to the relationship between the standard deviation of the sample mean and sample size that is discussed here in the textbook. It takes a sample size n (a number) as its argument. It returns the predicted value of the standard deviation of the mean delay time for samples of size n from the flight delays (represented in the table united).

# In[18]:


def predict_sample_mean_sd(n):
    means = make_array()
    
    for i in np.arange(10000):
        means = np.append(means, np.mean(delays.sample(n).column('Delay')))
    sd = np.std(means)
    return sd

predict_sample_mean_sd(10)


# In[19]:


predict_sample_mean_sd(500)


# >The predicted value of the standard deviation of the mean delay time for samples of size 10 was around 12.29 whereas the predicted valued of the same for sample size 500 was around 1.77. This reinforces the core idea of the standard deviation that as the sample size increases, the standard deviation tends to decrease.

# In[ ]:




