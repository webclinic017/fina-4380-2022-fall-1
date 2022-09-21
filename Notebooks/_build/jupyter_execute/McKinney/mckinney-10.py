#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Data Aggregation and Group Operations

# ## Introduction
# 
# Chapter 10 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses groupby operations, which are the pandas equivalent of Excel pivot tables.
# Pivot tables help us calculate statistics (e.g., sum, mean, and median) for one set of variables by groups of other variables (e.g., weekday or year).
# For example, we could use a pivot table to calculate mean daily stock returns by weekday.
# 
# We will focus on:
# 
# 1. Using `.groupby()` to group by columns, indexes, and functions
# 1. Using `.agg()` to aggregate multiple functions
# 1. Using pivot tables as an alternative to `.groupby()`
# 
# ***Note:*** Indented block quotes are from McKinney, and section numbers differ from McKinney because we will not discuss every topic.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


plt.rcParams['figure.dpi'] = 150
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import requests_cache
session = requests_cache.CachedSession(expire_after='1D')
import yfinance as yf
import pandas_datareader as pdr


# ## GroupBy Mechanics
# 
# "Split-apply-combine" is an excellent way to describe and visualize pandas groupby operations.
# 
# > Hadley Wickham, an author of many popular packages for the R programming 
# language, coined the term split-apply-combine for describing group operations. In the
# first stage of the process, data contained in a pandas object, whether a Series, DataFrame, or otherwise, is split into groups based on one or more keys that you provide.
# The splitting is performed on a particular axis of an object. For example, a DataFrame
# can be grouped on its rows (axis=0) or its columns (axis=1). Once this is done, a
# function is applied to each group, producing a new value. Finally, the results of all
# those function applications are combined into a result object. The form of the resulting object will usually depend on what’s being done to the data. See Figure 10-1 for a
# mockup of a simple group aggregation.
# 
# Figure 10-1 visualizes a pandas groupby operation that:
# 
# 1. Splits the dataframe by the `key` column (i.e., "groups by `key`")
# 2. Applies the sum operation to the `data` column (i.e., "sums `data`")
# 3. Combines the grouped sums
# 
# I describe this operation as "sum the `data` column by (gruops formed on) the `key` column."

# In[4]:


np.random.seed(42)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})


# Here is one way to calculate the means of `data1` by (grouops formed on) `key1`.

# In[5]:


df.loc[df['key1'] == 'a', 'data1'].mean()


# In[6]:


df.loc[df['key1'] == 'b', 'data1'].mean()


# We can do this calculation more quickly:
# 
# 1. Use the `.groupby()` method to group by `key1`
# 2. Use the `.mean()` method to sum `data1` within each value of `key1`
# 
# Note that without the `.mean()` method, pandas only sets up the grouped object, which can accept the `.mean()` method.

# In[7]:


grouped = df['data1'].groupby(df['key1'])


# In[8]:


grouped


# In[9]:


grouped.mean()


# We can can chain the `.groupby()` and `.mean()` methods.

# In[10]:


df['data1'].groupby(df['key1']).mean()


# If we prefer our result as a dataframe instead of a series, we can wrap `data1` with two sets of square brackets.

# In[11]:


df[['data1']].groupby(df['key1']).mean()


# We can group by more than one variable.
# We get a hierarchical row index (or row multi-index) when we group by more than one variable.

# In[12]:


means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means


# We can use the `.unstack()` method if we want to use both rows and columns to organize data.
# Note that, the `.unstack()` method un-stacks the last dimension (i.e., `level = -1`) by default so that `key2` values become columns.

# In[13]:


means.unstack()


# The grouping variables can also be columns in the data frame passed to the `.groupby()` method.
# I prefer this approach because we will typically have all data in one data frame.

# In[14]:


df.groupby('key1').mean()


# In[15]:


df.groupby(['key1', 'key2']).mean()


# There are many more methods than `.mean()`.
# We can use tab completion to discover (or remind ourselves of) these other methods.

# ### Iterating Over Groups
# 
# We can iterate over groups, too.
# The `.groupby()` method generates a sequence of tuples.
# Each tuples contains the value(s) of the grouping variable(s) and associated chunk of the dataframe.
# McKinney provides two loops to show how to iterate over groups.

# In[16]:


for name, group in df.groupby('key1'):
    print(name)
    print(group)


# In[17]:


for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)


# ### Selecting a Column or Subset of Columns
# 
# We preview the idea of grouping an entire dataframe above.
# However, I want to take this chance to explain McKinney's use of the phrase "syntactic sugar."
# Here is the context:
# 
# > Indexing a GroupBy object created from a DataFrame with a column name or array
# of column names has the effect of column subsetting for aggregation. This means
# that:
# >
# > ```
# > df.groupby('key1')['data1']
# > df.groupby('key1')[['data2']]
# > ```
# >
# > are syntactic sugar for
# >
# > ```
# > df['data1'].groupby(df['key1'])
# > df[['data2']].groupby(df['key1'])
# > ```
# 
# "Syntactic sugar" makes code easier to type or read without adding functionality.
# It makes code "sweeter" for humans to type or read by making it more concise or clear.
# The implication is that syntactic sugar makes code faster to type/read but does make code faster to execute.

# In[18]:


df.groupby(['key1', 'key2'])[['data2']].mean()


# ### Grouping with Functions
# 
# We can also group with functions.
# Below, we group with the `len` function, which calculates the length of the first names in the row index.
# We could instead add a helper column to `people`, but it is easier to pass a function to `.groupby()`.

# In[19]:


np.random.seed(42)
people = pd.DataFrame(
    data=np.random.randn(5, 5), 
    columns=['a', 'b', 'c', 'd', 'e'], 
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
)


# In[20]:


people.groupby(len).sum()


# We can mix functions, lists, etc. that we pass to `.groupby()`.

# In[21]:


key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# In[22]:


d = {'Joe': 'a', 'Jim': 'b'}
people.groupby([len, d]).min()


# ### Grouping by Index Levels
# 
# We can also group by index levels.
# We can specify index levels by either level number or name.

# In[23]:


columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]],
                                    names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)


# In[24]:


hier_df.groupby(level='cty', axis=1).count()


# In[25]:


hier_df.groupby(level='cty', axis='columns').count()


# In[26]:


hier_df.groupby(level='tenor', axis=1).count()


# ## Data Aggregation
# 
# Table 10-1 provides the optimized groupby methods:
# 
# - `count`: Number of non-NA values in the group
# - `sum`: Sum of non-NA values
# - `mean`: Mean of non-NA values
# - `median`: Arithmetic median of non-NA values
# - `std`, `var`: Unbiased (n – 1 denominator) standard deviation and variance
# - `min`, `max`: Minimum and maximum of non-NA values
# - `prod`: Product of non-NA values
# - `first`, `last`: First and last non-NA values
# 
# These optimized methods are fast and efficient, but pandas does not limit us to these methods.
# First, any series method is available.

# In[27]:


df.groupby('key1')['data1'].quantile(0.9)


# Second, we can write our own functions and pass them to the `.agg()` method.
# These functions should accept an array and returns a single value.

# In[28]:


def peak_to_peak(arr):
    return arr.max() - arr.min()


# In[29]:


df.groupby('key1')['data1'].agg(peak_to_peak)


# Some other methods work, too, even if they are do not aggregate an array to a single value.

# In[30]:


df.groupby('key1')['data1'].describe()


# ### Column-Wise and Multiple Function Application
# 
# The `.agg()` methods provides two more handy features:
# 
# 1. We can pass multiple functions to operate on all of the columns
# 2. We can pass specific functions to operate on specific columns

# ## Apply: General split-apply-combine
# 
# The `.agg()` method aggrates an array to a single value.
# We can use the `.apply()` method for more general calculations.
# 
# We can combine the `.groupby()` and `.apply()` methods to:
# 
# 1. Split a dataframe by grouping variables
# 2. Call the applied function on each chunk of the original dataframe
# 3. Recombine the output of the applied function

# In[31]:


def top(x, col, n=1):
    return x.sort_values(col).head(n)


# In[32]:


df.groupby('key1').apply(top, col='data1', n=2)


# In[33]:


df.groupby('key1').apply(top, col='data2', n=2)


# ## Pivot Tables and Cross-Tabulation
# 
# Above we manually made pivot tables with the `groupby()`, `.agg()`, `.apply()` and `.unstack()` methods.
# pandas provides a literal interpreation of Excel-style pivot tables with the `.pivot_table()` method and the `pandas.pivot_table()` function.
# These also provide row and column totals via "margins".
# It is worthwhile to read-through the `.pivot_table()` docstring several times.

# In[34]:


ind = (
    yf.download(tickers='^GSPC ^DJI ^IXIC ^FTSE ^N225 ^HSI', session=session)
    .rename_axis(columns=['Variable', 'Index'])
    .stack()
)


# The default aggregation function for `.pivot_table()` is `mean`.

# In[35]:


ind.loc['2015':].pivot_table(index='Index')


# We can use 
#     `values` to select specific variables, 
#     `pd.Grouper()` to sample different date windows, 
#     and 
#     `aggfunc` to select specific aggregation functions.

# In[36]:


(
    ind
    .loc['2015':]
    .reset_index()
    .pivot_table(
        values='Close',
        index=pd.Grouper(key='Date', freq='A'),
        columns='Index',
        aggfunc=['min', 'max']
    )
)


# ## Practice

# In[37]:


np.random.seed(42)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})


# ***Practice:***
# Calculate the means of columns `data1` and `data2` by `key1` and `key2`, and arrange the results so that values of `key1` are in the rows and values of `key2` are in the columns.

# ***Practice:***
# Replicate the previous practice exercise with `pd.pivot_table()` and test equality with `np.allclose()`.
# We will learn more about `pd.pivot_table()` at the end of this notebook, but we can give it a try now.

# In[38]:


np.random.seed(42)
people = pd.DataFrame(
    data=np.random.randn(5, 5), 
    columns=['a', 'b', 'c', 'd', 'e'], 
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
)


# ***Practice:***
# Calculate the sum of columns `a` through `e` by groups formed on the last letter in each name.
# *Hint:* use an anonymous (lambda) function.

# ***Practice:***
# Use the `.to_clipboard()` method to check your answer to the previous practice exercise.

# We need data for the following two practice exercises.
# We have to jump through some hoops with `pd.MultiIndex.from_product()` if we want to take full advantage of pandas multi indexes.

# In[39]:


faang = yf.download(tickers='META AAPL AMZN NFLX GOOG', session=session)
faang.columns.names = ['Variable', 'Ticker']
faang[pd.MultiIndex.from_product([['Return'], faang['Adj Close'].columns])] = faang['Adj Close'].pct_change()


# ***Practice:***
# For the FAANG stocks, calulate the mean and standard deviation of returns by ticker.

# ***Practice:***
# For the FAANG stocks, calulate the mean and standard deviation of returns and the maximum of closing prices by ticker.
# To do this, pass a dictionary where the keys are the column names and the values are lists of functions.

# ***Practice:***
# 
# 1. Download all available daily data for the S&P 500 ETF and Google stock (tickers SPY and GOOG)
# 2. Calculate daily returns
# 3. Calculate the volatility (standard deviation) of daily returns *every month* by combining `pd.Grouper()` and `.groupby()`)
# 4. Multiply by $\sqrt{252}$ to annualize these volatilities of daily returns
# 5. Plot these annualized volatilities

# ***Practice:***
# 
# 1. Download the daily factor data from Ken French's website
# 1. Calculate daily market returns by summing the market risk premium and risk-free rates (`Mkt-RF` and `RF`, respectively)
# 1. Calculate the volatility (standard deviation) of daily returns *every month* by combining `pd.Grouper()` and `.groupby()`)
# 1. Multiply by $\sqrt{252}$ to annualize these volatilities of daily returns
# 1. Plot these annualized volatilities
# 
# Is market volatility higher during wars?
# Consider the following dates:
# 
# 1. WWII: December 1941 to September 1945
# 1. Korean War: 1950 to 1953
# 1. Viet Nam War: 1959 to 1975
# 1. Gulf War: 1990 to 1991
# 1. War in Afghanistan: 2001 to 2021
