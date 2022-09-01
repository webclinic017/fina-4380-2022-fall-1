#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 8 - Data Wrangling: Join, Combine, and Reshape

# ## Introduction
# 
# Chapter 8 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) introduces a few important pandas concepts:
# 
# 1. Joining or merging is combining 2+ data frames on 1+ indices or columns into 1 data frame
# 1. Reshaping is rearranging data frames so it has fewer columns and more rows (wide to long) or more columns and fewer rows (long to wide); we can also reshape a series to a data frame and vice versa
# 
# We will focus on combining data with `pd.merge()` and `.join()` and reshaping data with `pd.stack()` and `pd.unstack()`.
# 
# ***Note:*** Indented block quotes are from McKinney, and section numbers differ from McKinney because we will not discuss every topic.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[11]:


plt.rcParams['figure.dpi'] = 150
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[12]:


import requests_cache
session = requests_cache.CachedSession(expire_after='1D')
import yfinance as yf
import pandas_datareader as pdr


# ## Hierarchical Indexing
# 
# Hierarchical indexing provides 2+ index levels on an axis.
# For example, we could index rows by stock ticker and date.
# Or we could index columns by variable name and stock ticker.
# Hierarchical indexing helps us work with higher-dimensional data in a lower-dimensional form.

# In[13]:


np.random.seed(42)
data = pd.Series(
    data=np.random.randn(9),
    index=[
        ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        [1, 2, 3, 1, 3, 1, 2, 2, 3]
    ]
)


# We can partially index this series to concisely subset data.

# In[14]:


data['b']


# In[15]:


data['b':'c']


# In[16]:


data.loc[['b', 'd']]


# We can also subset on the inner index level.

# In[17]:


data.loc[:, 2]


# Here `data` has a stacked format.
# For each outer index level (the letters), we have multiple observations based on the inner index level (the numbers).
# We can un-stack `data` to convert the inner index level to columns.

# In[18]:


data.unstack()


# We can create a data frame with hieracrhical indices or multi-indices on rows *and* columns.

# In[19]:


frame = pd.DataFrame(
    data=np.arange(12).reshape((4, 3)),
    index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
    columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']]
)


# In[20]:


frame


# We can add names to these multi-indices but names are not required.

# In[21]:


frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']


# In[22]:


frame


# Recall that `df[val]` selects the `val` column (or columns when `val` is a list).
# Here `frame` has a multi-index for the columns, so `frame['Ohio']` selects all columns with Ohio as the outer index level.

# In[23]:


frame['Ohio']


# If we want the inner level of the column index, we have to do a more work.

# In[24]:


frame.loc[:, (slice(None), 'Green')]


# Here `pd.IndexSlice[:, 'Green']` is an alternative to `(slice(None), 'Green')`.

# In[25]:


frame.loc[:, pd.IndexSlice[:, 'Green']]


# We can pass a tuple if we only want one column. 

# In[26]:


frame.loc[:, [('Ohio', 'Green')]]


# ### Reordering and Sorting Levels
# 
# We can swap index levels.
# There are `i` and `j` arguments, with defaults `i=-2` and `j=-1`.
# Therefore, by default, `.swaplevel()` swaps the two inner index levels.

# In[27]:


frame.swaplevel()


# In[28]:


frame.swaplevel('key1', 'key2')


# In[29]:


frame.swaplevel(axis=1)['Green'] # same data with loss of color index level


# We can also sort on an index (or list of indices).
# After we swap levels, we may want to sort our data.

# In[30]:


frame.sort_index(level=1)


# In[31]:


frame.sort_index(level='key2')


# In[32]:


frame.sort_index(level=[0, 1])


# We can chain these methods, too.

# In[33]:


frame.swaplevel(0, 1).sort_index(level=0)


# ### Indexing with a DataFrame's columns

# In[38]:


frame = pd.DataFrame({
    'a': range(7), 
    'b': range(7, 0, -1),
    'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
    'd': [0, 1, 2, 0, 1, 2, 3]
})


# We can change a data frame's index with the `.set_index()` and `.reset_index()` methods.
# These methods are useful is we want to use tickers (or other stock identifiers) as row indices (or remove them as indices to perform other operations).
# When we set a column as an index, pandas removes it as a column be default.

# In[40]:


frame2 = frame.set_index(['c', 'd'])


# However, we can change this default if we want to keep (or not drop) these columns.

# In[42]:


frame.set_index(['c', 'd'], drop=False)


# The `.reset_index()` method removes indices and adds them as columns, although we can drop them.
# Note that without an index, `frame2` has an integer index.

# In[43]:


frame2.reset_index()


# ## Combining and Merging Datasets
# 
# pandas provides several ways to combine and merge data.
# These several ways can create the same output with different syntaxes and defaults.
# If the data I want to combine have similar indices, I try the `.join()` method first.
# The other advantage of `.join()` is that it can combine more than two data frames at once.
# Otherwise, I try `pd.merge()` first or its equivalent method `.merge()`.
# `pd.merge()` is more general than `.join()`, so we will start with `pd.merge()`.
# 
# The [pandas website](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#) provides helpful visualizations.

# ### Database-Style DataFrame Joins
# 
# > Merge or join operations combine datasets by linking rows using one or more keys. These operations are central to relational databases (e.g., SQL-based). The merge function in pandas is the main entry point for using these algorithms on your data.
# 
# We will start with the `pd.merge()` syntax, but pandas also has `.merge()` and `.join()` methods.
# Learning these other syntaxes is easy once we understand `pd.merge()`'s syntax.

# In[34]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})


# The default `how` is `how='inner'`, so `pd.merge` keeps only rows that appear in both data frames.

# In[35]:


pd.merge(df1, df2)


# An outer merge keeps all rows.

# In[36]:


pd.merge(df1, df2, how='outer')


# A left merge keeps only rows that appear in the left data frame.

# In[37]:


pd.merge(df1, df2, how='left')


# By default, `pd.merge()` merges on all columns that appear in both data frames.
# 
# > `on` : label or list
#     Column or index level names to join on. These must be found in both
#     DataFrames. If `on` is None and not merging on indexes then this defaults
#     to the intersection of the columns in both DataFrames.
#     
# `key` is the only common column between `df1` and `df2`.
# We *should* specify keys with `on` to avoid unexpected results.
# We *must* specify keys with `left_on` and `right_on` if we do not have a common column.

# In[38]:


pd.merge(df1, df2, on='key')


# In[39]:


df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})


# In[40]:


# pd.merge(df3, df4) # this code fails/errors because there are not common columns


# In[41]:


pd.merge(df3, df4, left_on='lkey', right_on='rkey')


# Note that `pd.merge()` dropped row `c` from `df3` and row `d` from `df4` in the previous example.
# Rows `c` and `d` dropped because `pd.merge()` *inner* joins be default.
# An inner join keeps the intersection of the left and right data frame keys.
# Further, note that rows `a` and `b` from `df4` appear three times to match `df3`.
# 
# If we want to keep rows `c` and `d`, we can *outer* join `df3` and `df4` with `how='outer'`.
# Note that missing values become `NaN`.

# In[42]:


pd.merge(df1, df2, how='outer')


# > Many-to-many merges have well-defined, though not necessarily intuitive, behavior.

# In[43]:


df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
df2 = pd.DataFrame({'key': ['a', 'b', 'a', 'b', 'd'], 'data2': range(5)})


# In[44]:


pd.merge(df1, df2, on='key', how='left')


# > Many-to-many joins form the Cartesian product of the rows. Since there were three 'b' rows in the left DataFrame and two in the right one, there are six 'b' rows in the result. The join method only affects the distinct key values appearing in the result.

# In[45]:


pd.merge(df1, df2, how='inner')


# Be careful with many-to-many joins!
# In finance, we typically do not expect many-to-many joins because we expect at least one of the data frames to have unique observations.
# Note that pandas will not warn us if we accidentally perform a many-to-many join instead of a one-to-one or many-to-one join.
# 
# We can merge on more than one key.
# For example, we may merge two data sets on ticker-date pairs or industry-date pairs.

# In[46]:


left = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                     'key2': ['one', 'two', 'one'],
                     'lval': [1, 2, 3]})
right = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                      'key2': ['one', 'one', 'one', 'two'],
                      'rval': [4, 5, 6, 7]})


# In[47]:


pd.merge(left, right, on=['key1', 'key2'], how='outer')


# When there are overlapping column names, `pd.merge()` appends `_x` and `_y` to the left and right versions of the overlapping columns.

# In[48]:


pd.merge(left, right, on='key1')


# I typically specify suffixes to avoid later confusion.

# In[49]:


pd.merge(left, right, on='key1', suffixes=('_left', '_right'))


# I read the `pd.merge()` docstring whenever I am in doubt, which is frequently.
# ***Table 8-2*** lists `pd.merge()`'s arguments.
# 
# > - `left`: DataFrame to be merged on the left side.
# > - `right`: DataFrame to be merged on the right side.
# > - `how`: One of 'inner', 'outer', 'left', or 'right'; defaults to 'inner'.
# > - `on`: Column names to join on. Must be found in both DataFrame objects. If not specified and no other join keys given will use the intersection of the column names in left and right as the join keys.
# > - `left_on`: Columns in left DataFrame to use as join keys.
# > - `right_on`: Analogous to left_on for left DataFrame.
# > - `left_index`: Use row index in left as its join key (or keys, if a MultiIndex).
# > - `right_index`: Analogous to left_index.
# > - `sort`: Sort merged data lexicographically by join keys; True by default (disable to get better performance in some cases on large datasets).
# > - `suffixes`: Tuple of string values to append to column names in case of overlap; defaults to ('_x', '_y') (e.g., if 'data' in both DataFrame objects, would appear as 'data_x' and 'data_y' in result).
# > - `copy`: If False, avoid copying data into resulting data structure in some exceptional cases; by default always copies.
# > - `indicator`: Adds a special column _merge that indicates the source of each row; values will be 'left_only', 'right_only', or 'both' based on the origin of the joined data in each row.

# ### Merging on Index
# 
# We use the `left_index` and `right_index` arguments if we want to merge on row indices.

# In[50]:


left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])


# In[51]:


pd.merge(left1, right1, left_on='key', right_index=True, how='outer')


# The index arguments work for hierarchical indices (multi indices), too.

# In[52]:


lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                      'key2': [2000, 2001, 2002, 2001, 2002],
                      'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                      index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                             [2001, 2000, 2000, 2000, 2001, 2002]],
                      columns=['event1', 'event2'])


# In[53]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True)


# In[54]:


pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')


# In[55]:


left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=['a', 'c', 'e'],
                     columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=['b', 'c', 'd', 'e'],
                      columns=['Missouri', 'Alabama'])


# If we merge on both indices, we keep the index.

# In[56]:


pd.merge(left2, right2, how='outer', left_index=True, right_index=True)


# > DataFrame has a convenient join instance for merging by index. It can also be used to combine together many DataFrame objects having the same or similar indexes but non-overlapping columns.
# 
# So if we have matching indices on left and right, we can use `.join()` for a more compact notation than `pd.merge()`.

# In[57]:


left2.join(right2, how='outer')


# Note that the `.join()` method left joins by default.
# Because `.join()` joins on indices by default, it requires few arguments.
# Therefore, we can pass a list of data frames to `.join()`.

# In[58]:


another = pd.DataFrame(
    data=[[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
    index=['a', 'c', 'e', 'f'],
    columns=['New York', 'Oregon']
)


# In[59]:


left2.join([right2, another])


# In[60]:


left2.join([right2, another], how='outer')


# ### Concatenating Along an Axis
# 
# `pd.concat()` is a flexible way to combine data frames and series along either axis.
# From the [pandas homepage](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#concatenating-objects):
# 
# >The concat() function (in the main pandas namespace) does all of the heavy lifting of performing concatenation operations along an axis while performing optional set logic (union or intersection) of the indexes (if any) on the other axes. Note that I say "if any" because there is only a single possible axis of concatenation for Series.
# 
# `pd.concat()` would be most useful for combining either:
# 
# 1. A list of data frames with similar layouts
# 1. A list of series because series do not have `.join()` or `.merge()` methods
# 
# The first is handy if we have to read and combine a directory of .csv files.

# In[61]:


s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])


# In[62]:


s1


# In[63]:


s2


# In[64]:


s3


# In[65]:


pd.concat([s1, s2, s3])


# In[66]:


pd.concat([s1, s2, s3], axis=1)


# In[67]:


s4 = pd.concat([s1, s3])


# In[68]:


pd.concat([s1, s4], axis=1)


# In[69]:


pd.concat([s1, s4], axis=1, join='inner')


# In[70]:


result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])


# In[71]:


result.unstack()


# In[72]:


pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])


# In[73]:


df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])
df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'], columns=['three', 'four'])


# In[74]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'])


# In[75]:


pd.concat({'level1': df1, 'level2': df2}, axis=1)


# In[76]:


pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])


# ## Reshaping and Pivoting
# 
# Above, we briefly explore reshaping data with `.stack()` and `.unstack()`.
# Here we explore reshaping data more deeply.

# ### Reshaping with Hierarchical Indexing
# 
# Hierarchical indices (multi-indices) help reshape data.
# 
# > There are two primary actions:
# > - stack: This "rotates" or pivots from the columns in the data to the rows
# > - unstack: This pivots from the rows into the columns

# In[77]:


data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                    name='number'))


# In[78]:


data


# In[79]:


result = data.stack()


# In[80]:


result


# In[81]:


result.unstack()


# In[82]:


result.unstack(0)


# In[83]:


result.unstack('state')


# In[84]:


s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])


# In[85]:


data2


# In[86]:


data2.unstack()


# Un-stacking may introduce missing values.

# In[87]:


data2.unstack()


# By default, stacking drops missing values, so these two operations are invertable.

# In[88]:


data2.unstack().stack()


# However, we can keep missing values with `dropna=False`.

# In[119]:


data2.unstack().stack(dropna=False)


# In[89]:


df = pd.DataFrame({
    'left': result, 
    'right': result + 5
},
    columns=pd.Index(['left', 'right'], name='side')
)


# Note that, when we un-stack, the un-stacked level becomes the innermost level in the resulting index.

# In[90]:


df.unstack('state')


# In[91]:


df.unstack('state').stack('side')


# McKinney provides two more subsections on reshaping data with the `.pivot()` and `.melt()` methods.
# Unlike, the stacking methods, the pivoting methods can aggregate data and do not require an index.
# We will skip these additional aggregation methods for now.

# ## Practice

# In[4]:


np.random.seed(42)
data = pd.Series(
    data=np.random.randn(9),
    index=[
        ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        [1, 2, 3, 1, 3, 1, 2, 2, 3]
    ]
)


# ***Practice:***
# Above, we un-stacked the inner index level to columns.
# Instead, un-stack the outer index level to columns.

# ***Practice:***
# Download data from Yahoo Finance for AAPL, FB, and MSFT to a data frame named `stocks`.
# Then add daily returns for each stock as the percent change in the adjusted closing price, and call it `Returns`.
# Finally, wrangle `stocks` as follows and assign to a new data frame `stocks_long`.
# *Hint:* You may want to use `pd.MultiIndex()` to add multiple return columns at once:
# 
# 1. Reshape `stocks` from wide to long and assign to `stocks_long`
#     1. Columns should have a single name (Open, High, Low, Close, etc.) with the name Variable
#     1. Row should have a multi-index on tickers and dates, in that order, with the names Ticker and Date
# 1. Sort `stocks_long` so that data are in chronological order within alphabetical blocks by ticker

# ***Practice:***
# Remove the ticker-date index from the `stocks_long` data frame and add it as columns.
# Name this new data frame `stocks_long_2`.

# ***Practice:***
# Add back the ticker-date index from the `stocks_long` data frame.
# Name this new data frame `stocks_long_3`.
# Use `np.allclose()` to compare `stocks_long` and `stocks_long_3`.

# ***Practice:***
# Merge the Fama-French daily factor data to the `stocks` and `stocks_long` data frames from above.
# The Fama-French daily factor data are simple returns as percents, so divide by 100 to create decimal returns.
