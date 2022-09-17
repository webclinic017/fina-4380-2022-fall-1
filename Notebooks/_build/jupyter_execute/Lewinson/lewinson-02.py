#!/usr/bin/env python
# coding: utf-8

# # Lewinson Chapter 2 - Technical Analysis in Python

# ## Introduction
# 
# Chapter 2 of Eryk Lewinson's [*Python for Finance Cookbook*](https://www.packtpub.com/product/python-for-finance-cookbook/9781789618518) discusses a handful of trading strategies based on technical analysis.
# 
# We will focus on implementing and evaluating a trading strategy based on past prices and returns.
# 
# ***Note:*** Indented block quotes are from Lewinson, and section numbers differ from Lewinson because we will not discuss every topic.

# I will simplify and streamline his code, where possible.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


plt.rcParams['figure.dpi'] = 150
get_ipython().run_line_magic('precision', '')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# ## Backtesting a Strategy Based on Simple Moving Average
# 
# Lewinson uses the backtrader package to implement his technical trading strategies.
# However, there are no recent code commits to [backtrader's GitHub repository](https://github.com/mementum/backtrader), so it may be abandoned.
# So we manually implement these stategies and learn a lot in the process!
# 
# The first strategy is a simple moving average (SMA) stategy that:
# 
# > For this recipe, we consider a basic strategy based on the SMA. The key points of the strategy are as follows:
# >
# >   - When the close price becomes higher than the 20-day SMA, buy one share.
# >   - When the close price becomes lower than the 20-day SMA and we have a share, sell it.
# >   - We can only have a maximum of one share at any given time.
# >   - No short selling is allowed.
# 
# We do these calculations in *dollar* terms instead of *share* terms.

# In[ ]:


aapl = yf.download('AAPL', session=session)


# Second, we calculate daily returns and add SMA(20) for the adjusted close.
# We use the adjust close because we do not want to misinterpret splits and dividends as price changes.

# In[ ]:


aapl['AAPL'] = aapl['Adj Close'].pct_change()
aapl['SMA20'] = aapl['Adj Close'].rolling(20).mean()


# Third, we add a `Position` column based on AAPL's adjusted close and SMA(20) columns.
# `Position` takes one of two values: `1` if we are long AAPL and `0` if we are neutral AAPL.
# `np.select()` avoids nested `np.where()` and accepts a default.
# We `.shift()` inputs one day because we do not know closing prices and SMA(20) until the end of the day.
# Therefore, we cannot update `Position` until the next trading day.

# In[ ]:


aapl['Position'] = np.select(
    condlist=[
        aapl['Adj Close'].shift() > aapl['SMA20'].shift(), # .shift() to use lagged values to prevent look-ahead bias
        aapl['Adj Close'].shift() <= aapl['SMA20'].shift()
    ], 
    choicelist=[1, 0],
    default=np.nan
)


# I find the following two steps helpful.
# First, plot the adjusted close, SMA(20), and position for a short window.

# In[ ]:


aapl.loc['1981-01':'1981-02', ['Adj Close', 'SMA20', 'Position']].plot(secondary_y='Position')
plt.title('AAPL SMA(20) Strategy')
plt.show()


# Second, copy-and-paste these data to Excel!
# Excel is an excellent place to check your work!

# In[ ]:


# aapl.loc[:'1981-02'].to_clipboard()


# Finally, we create a `Strategy` column that provides the return on the strategy.
# We will assume that we earn a cash return of 0% when we are neutral AAPL.

# In[ ]:


aapl['Strategy'] = aapl['Position'] * aapl['AAPL']


# We can plot the cumulative return on 1 dollar invested in this SMA(20) strategy.
# We drop missing values to make an apples-to-apples comparison between the buy-and-hold and SMA(20) strategies.
# There may be missing values for both strategies because:
# 
# 1. We need 2 days to calculate 1 daily return 
# 1. We need 20 days to calculate the first SMA(20)

# In[ ]:


(
    aapl
    [['AAPL', 'Strategy']]
    .dropna()
    .add(1)
    .cumprod()
    .rename(columns={'Strategy': 'SMA(20)'})
    .plot()
)
buy_date = (
    aapl
    [['AAPL', 'Strategy']]
    .dropna()
    .index[0] - 
    pd.offsets.BDay(1)
)
plt.ylabel('Value ($)')
plt.title(
    'Value of $1 Invested\n' +
    'at Close Price on ' + buy_date.strftime('%B %d, %Y')
)
plt.show()


# ## Calculating Bollinger Bands and testing a buy/sell strategy
# 
# John Bollinger developed Bollinger Bands in the early 1980s.
# He describes Bollinger Bands as follows:
# 
# > So what are Bollinger Bands? They are curves drawn in and around the price structure usually consisting of a moving average (the middle band), an upper band, and a lower band that answer the question as to whether prices are high or low on a relative basis. Bollinger Bands work best when the middle band is chosen to reflect the intermediate-term trend, so that trend information is combined with relative price level data.
# 
# More [here](https://www.bollingerbands.com/bollinger-bands).
# John Bollinger provides a list of rules [here](https://www.bollingerbands.com/bollinger-band-rules).
# In short, Bollinger Bands are bands around a trend (typically $\mu_{price} \pm 2\sigma_{price}$ using 20 trading days).
# Technical analysts use these bands to signal high and low prices.
# 
# Lewinson builds Bollinger Bands with the backtrader package, but we will build Bollinger Bands with pandas.
# Lewinson desribes Bollinger Bands as follows:
# 
# > Bollinger Bands are a statistical method, used for deriving information about the prices and volatility of a certain asset over time. To obtain the Bollinger Bands, we need to calculate the moving average and standard deviation of the time series (prices), using a specified window (typically, 20 days). Then, we set the upper/lower bands at K times (typically, 2) the moving standard deviation above/below the moving average.
# >
# > The interpretation of the bands is quite sample: the bands widen with an increase in volatility and contract with a decrease in volatility.
# > 
# > In this recipe, we build a simple trading strategy, with the following rules:
# >
# >    - Buy when the price crosses the lower Bollinger Band upwards.
# >    - Sell (only if stocks are in possession) when the price crosses the upper Bollinger Band downward.
# >    - All-in strategy—when creating a buy order, buy as many shares as possible.
# >    - Short selling is not allowed.
# 
# We will implement Lewinson's strategy with Tesla.
# First, we will plot the 20-day rolling means and plus/minus 2 standard deviations.

# In[ ]:


tsla = yf.download('TSLA', session=session)


# In[ ]:


tsla['TSLA'] = tsla['Adj Close'].pct_change()


# In[ ]:


win = 20
K = 2
tsla[['SMA20', 'SMV20']] = tsla['Adj Close'].rolling(win).agg(['mean', 'std'])
tsla['LB20'] = tsla['SMA20'] - K*tsla['SMV20']
tsla['UB20'] = tsla['SMA20'] + K*tsla['SMV20']


# In[ ]:


tsla.loc['2020', ['Adj Close', 'LB20', 'UB20']].plot(style=['b-', 'g--', 'g--'])
plt.legend(['Adj Close', 'Bollinger Bands'])
plt.ylabel('Price ($)')
plt.title('TSLA Bollinger Band ({}, {}) Strategy'.format(win, K))
plt.show()


# We will implement the TSLA Bollinger Band (20, 2) strategy in class.

# ## Calculating the relative strength index and testing a long/short strategy
# 
# Lewinson describes the relative strength index (RSI) as follows:
# 
# > The RSI is an indicator that uses the closing prices of an asset to identify oversold/overbought conditions. Most commonly, the RSI is calculated using a 14-day period, and it is measured on a scale from 0 to 100 (it is an oscillator). Traders usually buy an asset when it is oversold (if the RSI is below 30), and sell when it is overbought (if the RSI is above 70). More extreme high/low levels, such as 80-20, are used less frequently and, at the same time, imply stronger momentum.
# > 
# > In this recipe, we build a trading strategy with the following rules:
# >
# >    - We can go long and short.
# >    - For calculating the RSI, we use 14 periods (trading days).
# >    - Enter a long position if the RSI crosses the lower threshold (standard value of 30) upwards; exit the position when the RSI becomes larger than the middle level (value of 50).
# >    - Enter a short position if the RSI crosses the upper threshold (standard value of 70) downwards; exit the position when the RSI becomes smaller than 50.
# >    - Only one position can be open at a time.
# 
# Lewinson uses a package to calculate the RSI and implement his strategy above.
# We do not need a package!
# Here is RSI's formula: $$RSI = 100 - \frac{100}{1 + RS},$$ where $$RS = \frac{SMA(U, n)}{SMA(D, n)}.$$
# For "up days", $U = \Delta Adj\ Close$ and $D = 0$.
# For "down days", $U = 0$ and $D = - \Delta Adj\ Close$, so that $U$ and $D$ are always non-negative.
# We can learn more about RSI [here](https://en.wikipedia.org/wiki/Relative_strength_index).
# 
# We will use Tesla data, again, for this section, but in a new data frame `tsla2`.

# In[ ]:


tsla2 = yf.download('TSLA', session=session)


# In[ ]:


tsla2['TSLA'] = tsla2['Adj Close'].pct_change()


# First, we will write a function `rsi()` that calculates $RSI$ for a return series.
# Here are some details:
# 
# 1. We will make `rsi()`'s accept a series `x`, which can be either a series of dollar changes or a series of simple returns
# 1. We will make `rsi()`'s default window `n=14`

# Second, we will use `rsi()` to implement the RSI(14) strategy for TSLA.

# ## Practice

# ***Practice:***
# Implement the SMA(20) strategy above with AAPL with one chained calculation.
# Save assign this new data frame to `aapl2`.

# ***Practice:***
# Use `np.allclose()` to compare `aapl` and `aapl2`.

# ***Practice:***
# What is the full-sample Sharpe Ratio for the SMA(20) strategy with AAPL?
# Use the risk-free rate `RF` from Ken French's daily factors.

# ***Practice:***
# The AAPL SMA(20) strategy outperformed buy-and-hold for the full sample.
# How do SMA(20) and buy-and-hold perform over rolling 6-month windows?
# Plot the values of $1 invested in each for all possible 126-trading day windows.

# ***Practice:***
# Implement the BB(20, 2) strategy above with TSLA with one chained calculation.
# Save assign this new data frame to `tsla3`.

# ***Practice:***
# What is the full-sample Sharpe Ratio for the BB(20, 2) strategy with TSLA?
# Use the risk-free rate `RF` from Ken French's daily factors.

# ***Practice:***
# Implement the RSI(14) strategy above with TSLA with one chained calculation.
# Save assign this new data frame to `tsla4`.

# ***Practice:***
# What is the full-sample Sharpe Ratio for the RSI(14) strategy with TSLA?
# Use the risk-free rate `RF` from Ken French's daily factors.
