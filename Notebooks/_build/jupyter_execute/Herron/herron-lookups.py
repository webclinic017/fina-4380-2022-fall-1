#!/usr/bin/env python
# coding: utf-8

# # Herron - Lookups

# I will not know the answer to every question in class.
# When I do not, I will research and answer them here!

# ## 2022-09-14 Wednesday

# ### What is exactly is an `f` string?
# 
# [An `f` string is a new and improved way to format strings in Python](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python).
# Say I want to print "The office number for `professor` is `office`", where `professor` is a professor's name and `office` is her office number.
# The old way is to either concatenate (add) strings or use the `.format()` method.

# In[1]:


professor = 'Richard Herron'
office = 'Hayden Hall 120C'


# Concatenate strings:

# In[2]:


print('The office number for ' + professor + ' is ' + office)


# Use the `.format()` method:

# In[3]:


print('The office number for {} is {}'.format(professor, office))


# The `f` string format is easier to read:

# In[4]:


print(f'The office number for {professor} is {office}')


# Note we need the `f` in front parse the `{professor}` and `{office}`.
# [Here](https://peps.python.org/pep-0498/) is a more complete explanation.
