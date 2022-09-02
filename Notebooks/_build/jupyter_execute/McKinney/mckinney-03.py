#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 3 - Built-in Data Structures, Functions, and Files

# ## Introduction
# 
# We must understand Python's core functionality to make full use of NumPy and pandas.
# Chapter 3 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses Python's core functionality.
# We will focus on the following:
# 
# 1. Data structures (we will ignore sets)
#     1. tuples
#     1. lists
#     1. dicts (also known as dictionaries)
# 1. List comprehensions (we will ignore set and dict comprehensions)
# 1. Functions
#     1. Returning multiple values
#     1. Using anonymous functions
# 
# ***Note:*** Block quotes are from McKinney, and section numbers differ from McKinney because we will not discuss every topic.

# ## Data Structures and Sequences
# 
# > Python's data structures are simple but powerful. Mastering their use is a critical part
# of becoming a proficient Python programmer.

# ### Tuple
# 
# > A tuple is a fixed-length, immutable sequence of Python objects.
# 
# We cannot change a tuple after we create it, so tuples are immutable.
# A tuple is ordered, so we can subset or slice it with a numerical index.
# We will surround tuples with parentheses but the parentheses are not always required.

# In[1]:


tup = (4, 5, 6) # the parentheses are optional, but we will use them


# In[2]:


# tup = 4, 5, 6


# In[3]:


tup[0] # PYTHON IS ZERO-INDEXED!


# In[4]:


nested_tup = ((4, 5, 6), (7, 8))


# In[5]:


nested_tup[0] # PYTHON IS ZERO-INDEXED!


# In[6]:


nested_tup[0][0]


# In[7]:


tuple([4, 0, 2])


# In[8]:


tup = tuple('string')


# In[9]:


tup[0]


# In[10]:


tup = tuple(['foo', [1, 2], True])


# In[11]:


# tup[2] = False # gives an error, because tuples are immutable (unchangeable)


# > If an object inside a tuple is mutable, such as a list, you can modify it in-place.

# In[12]:


tup


# > You can concatenate tuples using the + operator to produce longer tuples:
# 
# Tuples are immutable.
# So we cannot modify a tuple, but we can combine two tuples into a new tuple.

# In[13]:


(1, 2) + (1, 2)


# In[14]:


(4, None, 'foo') + (6, 0) + ('bar',)


# > Multiplying a tuple by an integer, as with lists, has the effect of concatenating together
# that many copies of the tuple:
# 
# This multiplication behavior is the logical extension of the addition behavior above.
# The output of `tup + tup` should be the same as the output of `2 * tup`.

# In[15]:


('foo', 'bar') * 4


# In[16]:


(1, 2) * 4


# #### Unpacking tuples
# 
# > If you try to assign to a tuple-like expression of variables, Python will attempt to
# unpack the value on the righthand side of the equals sign.

# In[17]:


tup = (4, 5, 6)
a, b, c = tup


# We can even unpack nested tuples!

# In[18]:


tup = 4, 5, (6, 7)
a, b, (c, d) = tup


# You can use this functionality to rename variables without a using a third, temporary variable.

# #### Tuple methods
# 
# > Since the size and contents of a tuple cannot be modified, it is very light on instance
# methods. A particularly useful one (also available on lists) is count, which counts the
# number of occurrences of a value.

# In[19]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# In[20]:


a.index(2)


# ### List
# 
# > In contrast with tuples, lists are variable-length and their contents can be modified in-place. You can define them using square brackets [ ] or using the list type function.

# In[21]:


a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)


# In[22]:


a_list


# In[23]:


b_list


# In[24]:


a_list[0] # PYTHON IS ZERO-INDEXED!


# #### Adding and removing elements
# 
# > Elements can be appended to the end of the list with the append method.
# 
# The `.append()` method appends an element to the list *in place* without reassigning the list.

# In[25]:


b_list.append('dwarf')


# > Using insert you can insert an element at a specific location in the list.
# The insertion index must be between 0 and the length of the list, inclusive.

# In[26]:


b_list.insert(1, 'red') # PYTHON IS ZERO INDEXED


# In[27]:


b_list


# In[28]:


b_list.index('red')


# In[29]:


b_list[b_list.index('red')] = 'blue'


# In[30]:


b_list


# > The inverse operation to insert is pop, which removes and returns an element at a
# particular index.

# In[31]:


b_list.pop(2)


# Note that `.pop(2)` removes the 2 element.
# If we do not want to remove the 2 element, we should use `[2]` to access an element without removing it.

# In[32]:


b_list


# > Elements can be removed by value with remove, which locates the first such value and removes it from the list.

# In[33]:


b_list.append('foo')


# In[34]:


b_list.remove('foo')


# In[35]:


b_list


# > If performance is not a concern, by using append and remove, you can use a Python list as a perfectly suitable "multiset" data structure.
# 
# However, appending to and removing from a list is very slow.
# When an example arises, we will benchmark appending to a list against alternatives.

# In[36]:


'dwarf' in b_list


# In[37]:


'dwarf' not in b_list


# #### Concatenating and combining lists
# 
# > Similar to tuples, adding two lists together with + concatenates them.

# In[38]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# In[39]:


xx = [4, None, 'foo']
xx.append([7, 8, (2, 3)])


# > If you have a list already defined, you can append multiple elements to it using the extend method.

# In[40]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])


# #### Sorting
# 
# > You can sort a list in-place (without creating a new object) by calling its sort function.

# In[41]:


a = [7, 2, 5, 1, 3]
a.sort()


# > sort has a few options that will occasionally come in handy. One is the ability to pass a secondary sort key—that is, a function that produces a value to use to sort the objects. For example, we could sort a collection of strings by their lengths.
# 
# Before you write your own solution to a problem, read the docstring (help file) of the built-in function.
# The built-in function may already solve your problem (typically faster and with fewer bugs).

# In[42]:


b = ['saw', 'small', 'He', 'foxes', 'six']


# In[43]:


b.sort()


# In[44]:


b.sort(key=len)


# #### Slicing
# 
# ***Slicing is very important!***
# 
# > You can select sections of most sequence types by using slice notation, which in its basic form consists of start:stop passed to the indexing operator [ ].
# 
# Recall that Python is zero-indexed, so the first element has an index of 0.
# The necessary consequence of zero-indexing is that start:stop is inclusive on the left edge (start) and exclusive on the right edge (stop).

# In[45]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]


# In[46]:


seq[:5]


# In[47]:


seq[1:5]


# In[48]:


seq[3:5]


# > Either the start or stop can be omitted, in which case they default to the start of the sequence and the end of the sequence, respectively.

# In[49]:


seq[:5]


# In[50]:


seq[3:]


# > Negative indices slice the sequence relative to the end.

# In[51]:


seq


# In[52]:


seq[-4:]


# In[53]:


seq[-4:-1]


# In[54]:


seq[-6:-2]


# > A step can also be used after a second colon to, say, take every other element.

# In[55]:


seq


# In[56]:


seq[::2]


# In[57]:


seq[1::2]


# I remember the trick above as `:2` is "count by 2".

# > A clever use of this is to pass -1, which has the useful effect of reversing a list or tuple.

# In[58]:


seq[::-1]


# We will use slicing (subsetting) all semester, so it is worth a few minutes to understand the examples above.

# ### dict
# 
# > dict is likely the most important built-in Python data structure. A more common
# name for it is hash map or associative array. It is a flexibly sized collection of key-value
# pairs, where key and value are Python objects. One approach for creating one is to use
# curly braces {} and colons to separate keys and values.
# 
# Elements in dictionaries have names, while elements in tuples and lists have numerical indices.
# Dictionaries are handy for passing named arguments and returning named results.

# In[59]:


empty_dict = {}


# A dictionary is a set of key-value pairs.

# In[60]:


d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}


# In[61]:


d1['a']


# In[62]:


d1[7] = 'an integer'


# We access dictionary elements by name/label instead of position.

# In[63]:


d1['b']


# In[64]:


'b' in d1


# > You can delete values either using the del keyword or the pop method (which simultaneously returns the value and deletes the key).

# In[65]:


d1[5] = 'some value'


# In[66]:


d1['dummy'] = 'another value'


# In[67]:


d1


# In[68]:


del d1[5]


# In[69]:


d1


# In[70]:


ret = d1.pop('dummy')


# In[71]:


ret


# In[72]:


d1


# > The keys and values method give you iterators of the dict’s keys and values, respectively. While the key-value pairs are not in any particular order, these functions output the keys and values in the same order.

# In[73]:


d1.keys()


# In[74]:


d1.values()


# > You can merge one dict into another using the update method.

# In[75]:


d1.update({'b' : 'foo', 'c' : 12})


# In[76]:


d1


# We will skip the rest of the dictionary topics from McKinney, as we are unlikely to use them.

# ## List, Set, and Dict Comprehensions
# 
# > List comprehensions are one of the most-loved Python language features. They allow you to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter in one concise expression. They take the basic form:
# > ```python
# > [expr for val in collection if condition]
# > ```
# > This is equivalent to the following for loop:
# > ```python
# > result = []
# > for val in collection:
# >     if condition:
# >     result.append(expr)
# > ```
# > The filter condition can be omitted, leaving only the expression.
# 
# We will focus on list comprehensions, which are very [Pythonic](https://blog.startifact.com/posts/older/what-is-pythonic.html).

# In[77]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']


# In[78]:


caps = []
for x in strings:
    if len(x) > 2:
        caps.append(x.upper())
caps


# We can replace the for loop above with a list comprehension.

# In[79]:


# [operation on x for x in list if condition is met]
[x.upper() for x in strings if len(x) > 2]


# ## Functions
# 
# > Functions are the primary and most important method of code organization and reuse in Python. As a rule of thumb, if you anticipate needing to repeat the same or very similar code more than once, it may be worth writing a reusable function. Functions can also help make your code more readable by giving a name to a group of Python statements.
# >
# > Functions are declared with the def keyword and returned from with the return keyword:
# > ```python
# > def my_function(x, y, z=1.5):
# >     if z > 1:
# >          return z * (x + y)
# >      else:
# >          return z / (x + y)
# > ```
# > There is no issue with having multiple return statements. If Python reaches the end of a function without encountering a return statement, None is returned automatically.
# >
# > Each function can have positional arguments and keyword arguments. Keyword arguments are most commonly used to specify default values or optional arguments. In the preceding function, x and y are positional arguments while z is a keyword argument. This means that the function can be called in any of these ways:
# > ```python
# >  my_function(5, 6, z=0.7)
# >  my_function(3.14, 7, 3.5)
# >  my_function(10, 20)
# > ```
# > The main restriction on function arguments is that the keyword arguments must follow the positional arguments (if any). You can specify keyword arguments in any order; this frees you from having to remember which order the function arguments were specified in and only what their names are.
# 
# Here is the basic syntax for a function:

# In[80]:


def mult_by_two(x):
    return 2*x


# ### Returning Multiple Values
# 
# We can write Python functions that return multiple objects.
# In reality, the function `f()` below returns one object, a tuple, that we can unpack to multiple objects.

# In[81]:


def f():
    a = 5
    b = 6
    c = 7
    return a, b, c


# If we want to return multiple objects with names or labels, we can return a dictionary.

# In[82]:


def f():
    a = 5
    b = 6
    c = 7
    return {'a' : a, 'b' : b, 'c' : c}


# In[83]:


f()['a']


# ### Anonymous (Lambda) Functions
# 
# > Python has support for so-called anonymous or lambda functions, which are a way of writing functions consisting of a single statement, the result of which is the return value. They are defined with the lambda keyword, which has no meaning other than "we are declaring an anonymous function."
# 
# > I usually refer to these as lambda functions in the rest of the book. They are especially convenient in data analysis because, as you'll see, there are many cases where data transformation functions will take functions as arguments. It's often less typing (and clearer) to pass a lambda function as opposed to writing a full-out function declaration or even assigning the lambda function to a local variable.
# 
# Lambda functions are very Pythonic and let us to write simple functions on the fly.
# For example, we could use a lambda function to sort `strings` by the number of unique letters.

# In[84]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[85]:


strings.sort(key=lambda x: len(set(list(x))))


# In[86]:


strings.sort(key=lambda x: x[-1]) # this lambda function slices the last character in each string i


# ## Practice

# ***Practice:***
# Swap the values assigned to `a` and `b` using a third variable `c`.

# In[87]:


a = 1


# In[88]:


b = 2


# ***Practice:***
# Use tuple-unpacking to swap the values assigned to `a` and `b` ***without*** using a third variable.

# ***Practice:*** 
# Create a list of integers from 1 to 100 using `range()` and `list()` named `l1`.

# ***Practice:*** 
# Slice `l1` to create a list of integers from 60 to 50 (inclusive) named `l2`.

# ***Practice:***
# Create a list of odd integers from 1 to 21 named `l3`.

# ***Practice:***
# Use a list comprehension to create a list `l4` that contains the squares of integers from 1 to 100.

# ***Practice:***
# Use a list comprehension to create a list `l5` that contains the squares of ***odd*** integers from 1 to 100.

# ***Practice:***
# Use a lambda function to sort `strings` by the last letter.

# You can find many more Python coding challenges at <www.hackerrank.com>.
# These challenges are fun but not necessary for our course because we will spend our time using higher-level features.
