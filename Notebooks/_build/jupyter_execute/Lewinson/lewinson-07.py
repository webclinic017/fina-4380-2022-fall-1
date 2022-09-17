#!/usr/bin/env python
# coding: utf-8

# # Lewinson Chapter 7 - Asset Allocation in Python

# ## Introduction
# 
# Chapter 7 of Eryk Lewinson's [*Python for Finance Cookbook*](https://www.packtpub.com/product/python-for-finance-cookbook/9781789618518) covers portfolios returns and optimization.
# 
# We will focus on:
# 
# 1. Evaluating $\frac{1}{n}$ portfolios (i.e., equal-weighted portfolios)
# 1. Using SciPy's optimizer to find the efficient frontier
# 1. Using SciPy's optimizer to achieve any objective
# 
# ***Note:*** Indented block quotes are from Lewinson, and section numbers differ from Lewinson because we will not discuss every topic.
# 
# I will simplify and streamline his code, where possible.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


plt.rcParams['figure.dpi'] = 150
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# ## Evaluating the performance of a basic 1/n portfolio
# 
# > We begin with inspecting the most basic asset allocation strategy: the 1/n portfolio. The idea is to assign equal weights to all the considered assets, thus diversifying the portfolio. As simple as that might sound, DeMiguel, Garlappi, and Uppal (2007) show that it can be difficult to beat the performance of the 1/n portfolio by using more advanced asset allocation strategies.
# >
# > The goal of the recipe is to show how to create a 1/n portfolio, calculate its returns, and then use a Python library called pyfolio to quickly obtain all relevant portfolio evaluation metrics in the form of a tear sheet. Historically, a tear sheet is a concise, usually one-page, document, summarizing important information about public companies.
# 
# We will not use `pyfolio`, which appears to be abandoned.
# 
# We will follow Lewinson's structure, although I prefer to download all data.
# These data are small (kilobytes instead of megabytes or gigabytes), so we might as well download all data then subset when necessary.

# In[ ]:


RISKY_ASSETS = ['AAPL', 'IBM', 'MSFT', 'TWTR']
START_DATE = '2017-01-01'
END_DATE = '2018-12-31'


# In[ ]:


df = yf.download(tickers=RISKY_ASSETS, session=session)


# In[ ]:


returns = df['Adj Close'].pct_change().loc[START_DATE:END_DATE]


# Before we do any portfolio math, we can make $\frac{1}{n}$ portfolios "by hand".

# In[ ]:


p1 = 0.25*returns['AAPL'] + 0.25*returns['IBM'] + 0.25*returns['MSFT'] + 0.25*returns['TWTR']


# In[ ]:


p2 = 0.25 * returns.sum(axis=1)


# In[ ]:


np.allclose(p1, p2)


# In[ ]:


p3 = returns.mean(axis=1)


# In[ ]:


np.allclose(p1, p3)


# ***Note that when we apply the same portfolio weights every day, we rebalance at the same frequency as the returns data.***
# If we have daily data, rebalance daily.
# If we have monthly data, we rebalance monthly, and so on.

# In[ ]:


returns.shape


# In[ ]:


portfolio_weights = np.ones(returns.shape[1]) / returns.shape[1]


# In[ ]:


portfolio_weights


# I prefer a different notation to calculate returns.
# The following notation is more compact than Lewinson's notation and generates the same output.
# $$R_P = \omega^T R,$$ where $R_P$ is a vector of portfolio returns, $\omega$ is a vector of portfolio weights, and $R$ is a matrix of individual stock or asset returns.

# In[ ]:


portfolio_returns = returns.dot(portfolio_weights)


# In[ ]:


np.allclose(p1, portfolio_returns)


# Here are some silly data to help understand the `.dot()` method.

# In[ ]:


silly = pd.DataFrame(np.arange(8).reshape(2, 4))


# In[ ]:


silly


# In[ ]:


silly.dot(portfolio_weights)


# In[ ]:


silly_weights = np.array([1, 0, 0, 0])


# In[ ]:


silly.dot(silly_weights)


# ## Finding the Efficient Frontier using Monte Carlo simulations
# 
# > According to the Modern Portfolio Theory, the Efficient Frontier is a set of optimal portfolios in the risk-return spectrum. This means that the portfolios on the frontier:
# > 
# > - Offer the highest expected return for a given level of risk
# > - Offer the lowest level of risk for a given level of expected returns
# > 
# > All portfolios located under the Efficient Frontier curve are considered sub-optimal, so it is always better to choose the ones on the frontier instead.
# > 
# > In this recipe, we show how to find the Efficient Frontier using Monte Carlo simulations. We build thousands of portfolios, using randomly assigned weights, and visualize the results. To do so, we use the returns of four US tech companies from 2018.
# 
# We will skip Lewinson's Monte Carlo simulation of the efficient frontier.
# For some students, this simulation is enlightening.
# However, for many students, it creates bad habits.
# Therefore, we will jump right to using the optimizer to find the efficient frontier.

# ## A Crash Course in scipy's minimize() Function

# The `sco.minimize()` function iteratively finds the input array `x` that minimizes the output of function `fun`.
# We pass our first guess for the input array `x` to the argument `x0=`.
# We pass additional arguments for the function `fun` as a tuple to the argument `args=`.
# We pass the lower and upper bounds on `x` as a tuple of tuples to the argument `bounds=`.
# We contrain our results with a tuple of dictionaries of functions to the argument`contraints=`.
# 
# Here is a simple example that minimizes the `quadratic()` function $y = (x - a)^2$ where $a=5$.

# In[ ]:


import scipy.optimize as sco


# In[ ]:


def quadratic(x, a=5):
    return (x - a) ** 2


# The minimum output of `quadratic()` occurs at $x=5$ if we do not use bounds or constraints.

# In[ ]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001])
)


# The minimum output of `quadratic()` occurs at $x=6$ if we bound `x` between 6 and 10 (i.e., $6 \leq x \leq 10$).

# In[ ]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001]),
    bounds=((6, 10),)
)


# Finally, the minimum output of `quadratic()` occurs at $x=6$, again, if bound `x` between 5 and 10 and we constrain `x - 6` to be positive.
# We use bounds to simply limit the search space, and we use constraints when we need to limit the search based on some formula.

# In[ ]:


sco.minimize(
    fun=quadratic,
    x0=np.array([2001]),
    bounds=((5, 10),),
    constraints=({'type': 'ineq', 'fun': lambda x: x - 6})
)


# Finally, we can use `sco.minimize()`s `args=` argument to to change the `a=` argument in `quadratic()`.
# Note that `args=` expects a tuple, so we need a trailing comma `,` if we only have one argument.

# In[ ]:


sco.minimize(
    fun=quadratic,
    args=(20,),
    x0=np.array([2001]),
)


# ## Finding the Efficient Frontier using optimization with scipy
# 
# > In the previous recipe, Finding the Efficient Frontier using Monte Carlo simulations, we used a brute-force approach based on Monte Carlo simulations to visualize the Efficient Frontier. In this recipe, we use a more refined method to determine the frontier.
# > 
# > From its definition, the Efficient Frontier is formed by a set of portfolios offering the highest expected portfolio return for a certain volatility, or offering the lowest risk (volatility) for a certain level of expected returns. We can leverage this fact, and use it in numerical optimization. The goal of optimization is to find the best (optimal) value of the objective function by adjusting the target variables and taking into account some boundaries and constraints (which have an impact on the target variables). In this case, the objective function is a function returning portfolio volatility, and the target variables are portfolio weights.
# > 
# > Mathematically, the problem can be expressed as: $$\min \omega^T \Sigma \omega$$ s.t. $$\omega^T \textbf{1} = 1$$ $$\omega \geq 0$$ $$\omega^T \mu = \mu_P$$
# >
# > Here, $\omega$ is a vector of weights, $\Sigma$ is the covariance matrix, $\mu$ is a vector of returns, $\mu_P$ and is the expected portfolio return.
# > 
# > We iterate the optimization routine used for finding the optimal portfolio weights over a range of expected portfolio returns, and this results in the Efficient Frontier.
# 
# We optimize our portfolios (i.e., minimize portfolio variance subject to contraints) with SciPy's optimize module.
# However, we will not use Lewinson's helper functions.
# Lewinson's approach is computationally efficient, but requires us to manage different $\mu$ (mean return vector) and $\Sigma$ (variance-covariance matrix) for every sample we want to consider.
# Instead, we will base our helper functions on weights and returns only.

# In[ ]:


def port_std(w, rs):
    return np.sqrt(252) * rs.dot(w).std()


# In[ ]:


res_mv = sco.minimize(
    fun=port_std, # we want to minimize portfolio volatility
    x0=np.ones(returns.shape[1]) / returns.shape[1], # x0 contains our first guess at portfolio weights
    args=(returns, ), # args provides additional argument to the function we will minimize
    bounds=((0, 1), (0, 1), (0, 1), (0, 1)), # bounds limits the search space for portfolio weights
    constraints=(
        {'type': 'eq', 'fun': lambda x: x.sum() - 1} # "eq" constraints are driven to zero
    )
)
assert res_mv['success']


# In[ ]:


res_mv


# What are the attributes of this minimum variance portfolio?

# In[ ]:


def print_port_res(w, title, df=returns):
    print(
        title,
        '=' * len(title),
        '',
        'Performance',
        '-----------',
        'Return:       {:0.4f}'.format(252 * df.dot(w).mean()),
        'Volatility:   {:0.4f}'.format(np.sqrt(252) * df.dot(w).std()),
        '',
        sep='\n'
    )

    print(
        'Weights', 
        '-------', 
        sep='\n'
    )
    for i, j in zip(df.columns, w):
        print((i + ':').ljust(14) + '{:0.4f}'.format(j))


# In[ ]:


print_port_res(res_mv['x'], 'Minimum Variance Portfolio')


# Now we will use these skills to map the efficient frontier in class.
# Here are some tips:
# 
# 1. Loop over a set of target returns from the best to worst
# 1. Use each target return as a constraint to `sco.minimize()`
# 1. Save your portfolio weights, returns, and volatilities to a data frame named `ef`

# ## Practice

# ***Practice:***
# Write functions for the following performance measures that Lewinson discusses:
# 
# > - Sharpe ratio: One of the most popular performance evaluation metrics, it measures the excess return (over the risk-free rate) per unit of standard deviation. When no risk-free rate is provided, the default assumption is that it is equal to 0%. The greater the Sharpe ratio, the better the portfolio's risk-adjusted performance.
# > - Max drawdown: A metric of the downside risk of a portfolio, it measures the largest peak-to-valley loss (expressed as a percentage) during the course of the investment. The lower the maximum drawdown, the better.
# > - Calmar ratio: The ratio is defined as the average annual compounded rate of return divided by the maximum drawdown for that same time period. The higher the ratio, the better.
# > - Stability: Measured as the R-squared of a linear fit to the cumulative log returns. In practice, this means regressing a range of integers (serving as the time index) on cumulative log returns.
# > - ~~Omega ratio: The probability-weighted ratio of gains over losses for a determined return target threshold (default set to 0). Its main advantage over the Sharpe ratio is that the Omega ratio—by construction—considers all moments of the returns distribution, while the former only considers the first two (mean and variance).~~
# > - Sortino ratio: A modified version of the Sharpe ratio, where the standard deviation in the denominator is replaced with downside deviation.
# > - Skew: Skewness measures the degree of asymmetry, that is, how much is the given distribution (here, of portfolio returns) more skewed than the Normal distribution. Negative skewness (left-skewed distributions) means  that large negative returns occur more frequently than large positive ones.
# > - Kurtosis: Measures extreme values in either of the tails. Distributions with large kurtosis exhibit tail data exceeding the tails of the Gaussian distribution, meaning that large and small returns occur more frequently.
# > - Tail ratio: The ratio (absolute) between the 95th and 5th percentile of the daily returns. A tail ratio of ~0.8 means that losses are ~1.25 times as bad as profits.
# > - Daily value at risk: Calculated as $\mu - 2\sigma$, where $\mu$ is the average portfolio return over the period, and $\sigma$ the corresponding standard deviation.
# 
# Here are some tips:
# 
# 1. Write functions that return decimal returns
# 1. For performance measures with benchmarks or targets, set the default to the risk-free rate of return from Ken French
# 1. Call these functions the lower-case version of the entry name with underscores instead of spaces
# 
# Also add:
# 
# 1. Total return
# 1. Annual geometric mean return
# 1. Annual volatility

# ***Practice:***
# Write a `tear_sheet()` function that tabulates average annual returns, cumulative returns, annual volatility, and the performance measures in the previous practice.

# ***Practice:***
# Find the portfolio with the maximum Sharpe Ratio for a portfolio of FAANG stocks from 2010 to 2019.
# Note that `sco.minimize()` finds *minimums*, so you need to minimize the *negative* Sharpe Ratio.

# ***Practice:***
# What is the *out-of-sample* performance of this maximum Sharpe Ratio portfolio?
# For example, what is the Sharpe Ratio of this portfolio from 2020 through today?
# How does this compare to the Sharpe Ratio of the $1/N$ portfolio?

# ***Practice:***
# Find the portfolio with the maximum Sharpe Ratio for a portfolio of FAANG stocks from 2010 to 2019, but allow short positions up to 30% of the portfolio.
# So for every one dollar invested, you can short one or more of these four stocks to finance another 30 cents.
