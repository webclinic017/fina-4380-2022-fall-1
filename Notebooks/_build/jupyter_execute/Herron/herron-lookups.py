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

# ## 2022-09-16 Friday

# ### Can we add attributes to a NumPy array?

# Yes!
# We can add attributes to a NumPy array: <https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray>.
# However, we will not add attributes to NumPy arrays or pandas data frames and series.
# Nor will we write our own classes because array, data frames, and series are feature complete for 99% of data analysis work.
# We would need to write our own classes if we wrote an econometrics package, but that is beyond this course and 99% of data analysis work.

# ### Are list comprehensions more than syntactic sugar?

# [Syntactic sugar](https://en.wikipedia.org/wiki/Syntactic_sugar) is:
# 
# > ...syntax within a programming language that is designed to make things easier to read or to express. It makes the language "sweeter" for human use: things can be expressed more clearly, more concisely, or in an alternative style that some may prefer. Syntactic sugar is usually a shorthand for a common operation that could also be expressed in an alternate, more verbose, form: The programmer has a choice of whether to use the shorter form or the longer form, but will usually use the shorter form since it is shorter and easier to type and read. 
# 
# I do not expect list comprehensions to be faster than for-loops, but it is hard to generalize and we would need to benchmark (`%%timeit`) specific cases.
# However, I consider the advantage of list comprehensions to be fast coding and comprehending instead of fast execution.
# [Here](https://realpython.com/list-comprehension-python/#benefits-of-using-list-comprehensions) is a clear discussion of the benefits of list comprehensions.
