#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 2 - Python Language Basics, IPython, and Jupyter Notebooks

# ## Introduction
# 
# We must understand the Python language basics before we can use it to analyze financial data.
# Chapter 2 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses Python, IPython, and Jupyter and the basics of the Python programming language.
# We will focus on the "Python Language Basics" in section 2.3, which covers language semantics, scalar types, and control flow.
# 
# ***Note:*** Block quotes are from McKinney, and our section numbers differ from McKinney because we will not discuss every topic.

# ## Language Semantics

# ### Indentation, not braces
# 
# > Python uses whitespace (tabs or spaces) to structure code instead of using braces as in many other languages like R, C++, Java, and Perl.
# 
# ***So, spaces are more than cosmetic in Python.***
# For non-Python programmers, white space is often Python's defining feature.

# In[1]:


array = [1, 2, 3] # list with three values
pivot = 2 # integer scalar
less = [] # empty list
greater = [] # empty list

for x in array:
    if x < pivot:
        print(x, 'is less than', pivot)
        less.append(x)        
    else:
        print(x, 'is NOT less than', pivot)
        greater.append(x)


# In[2]:


less


# In[3]:


greater


# Is 0 less than 2?

# In[4]:


less.append(0)


# In[5]:


less


# ### Comments
# 
# > Any text preceded by the hash mark (pound sign) # is ignored by the Python interpreter. This is often used to add comments to code. At times you may also want to exclude certain blocks of code without deleting them.
# 
# We can enter the hash mark anywhere in the line to ignore all code after the hash mark on the same line.
# We can quickly comment/un-comment code with `<Ctrl>-/` in notebook cells.

# In[6]:


# 5 + 5


# ### Function and object method calls
# 
# > You call functions using parentheses and passing zero or more arguments, optionally assigning the returned value to a variable:
# > ```python
# >     result = f(x, y, z)
# >     g()
# > ```
# > Almost every object in Python has attached functions, known as methods, that have access to the object's internal contents. You can call them using the following > syntax:
# > ```python
# >     obj.some_method(x, y, z)
# > ```
# > Functions can take both positional and keyword arguments:
# > ```python
# >     result = f(a, b, c, d=5, e='foo')
# > ```
# > More on this later.
# 
# We can write a function that adds two numbers.

# In[7]:


def add_numbers(a, b):
    return a + b


# In[8]:


add_numbers(5, 5)


# We can write a function that adds two strings, separated by a space.

# In[9]:


def add_strings(a, b):
    return a + ' ' + b


# In[10]:


add_strings('5', '5')


# What is the difference between `print()` and `return`?
# `print()` just prints to the console, while `return` lets us assign values.
# In the example below, we use the `return` line to assign the output of `add_string_2()` to the variable `return_from_add_strings_2`, and we use the `print()` line to print to the output cell, which is neither assigned nor captured.

# In[11]:


def add_strings_2(a, b):
    print(a + ' ' + b + ' (this is from the print statement)')
    return a + ' ' + b


# In[12]:


return_from_add_strings_2 = add_strings_2('5', '5')


# In[13]:


return_from_add_strings_2


# ### Variables and argument passing
# 
# > When assigning a variable (or name) in Python, you are creating a reference to the object on the righthand side of the equals sign.

# In[14]:


a = [1, 2, 3]


# In[15]:


a


# If we assign `a` to a new variable `b`, both `a` and `b` refer to the *same* object.
# This same object is the original list `[1, 2, 3]`.
# If we change `a`, we also change `b`, because both refer to the *same* object.

# In[16]:


b = a


# In[17]:


b


# ***Here, `a` and `b` are the same object with different names!***

# In[18]:


a is b


# In[19]:


a.append(4)


# In[20]:


a


# In[21]:


b


# In[22]:


b.append(5)


# In[23]:


a


# In[24]:


b


# When we append to `a`, we also appead to `b` because `a` and `b` are the same object!
# `a` and `b` are different names for the same object!
# This behavior is useful but a double-edged sword!
# [Here](https://nedbatchelder.com/text/names.html) is a deeper discussion of this behavior if you are curious.

# ### Dynamic references, strong types
# 
# > In contrast with many compiled languages, such as Java and C++, object references in Python have no type associated with them.
# 
# In Python, 
# 
# 1. We do not declare variables and their types
# 1. We can change variables' types because variables are only names that refer to objects

# In[25]:


b


# In[26]:


type(a)


# In[27]:


a = 5
type(a)


# In[28]:


a = 'foo'
type(a)


# However, Python is *strongly typed* and will only implicitly convert types in a few cases.
# For examples, in many programming languages, the following code returns either string '55' or integer 10.
# However, in Python, it returns an error.

# In[29]:


# '5' + 5


# A counterexample is that Python converts integers to floats.

# In[30]:


a = 4.5
b = 2
print('a is {0}, b is {1}'.format(type(a), type(b)))
a / b


# In the previous code cell:
# 
# 1. The output of `a / b` prints (or displays) because it is the last line in the code cell
# 1. However, if we assign this output to a variable (`c = a / b`), it will not print
# 1. The "a is ..." output prints because of the explicit `print()` function call
# 
# If we want integer division (or floor division) we have to use `//`.

# In[31]:


5 / 2


# In[32]:


5 // 2


# ### Attributes and methods
# 
# We can use tab completion to list attributes (characteristics stored inside objects) and methods (functions associated with objects).

# In[33]:


a = 'foo'


# In[34]:


a.capitalize()


# ### Imports
# 
# > In Python a module is simply a file with the .py extension containing Python code.
# 
# We can import a module with `import`. 
# There a few syntaxes for `import`ing modules.
# 
# If we `import pandas`, we have to use the full `pandas.` prefix

# In[35]:


import pandas


# The `import as ` syntax lets us define an abbreviated prefix.

# In[36]:


import pandas as pd


# We can also import one or more functions from a package with the following syntax.

# In[37]:


from pandas import DataFrame


# In[38]:


from pandas import DataFrame as df


# We will sometimes use the first syntax for lightly used packages and functions.
# We will most often use the second syntax with abbreviated prefixes to speed up coding and improve readability.
# We will sometimes use the third syntax for heavily used functions nested several modules deep.

# ### Binary operators and comparisons
# 
# Binary operators work like we expect based on our Excel experience.

# In[39]:


5 - 7


# In[40]:


12 + 21.5


# In[41]:


5 <= 2


# The textbook presents one way to make sure different names refer to different objects: perform any operation during assignment.

# In[42]:


a = [1, 2, 3]
b = a
c = list(a)


# In[43]:


a is b


# In[44]:


a is c


# Here `a` and `c` have the same *values*, but they are not the same object!

# In[45]:


a == c


# In[46]:


a is not c


# In Python, `=` is the assignment operator, `==` tests equality, and `!=` tests inequality.

# In[47]:


a == c


# In[48]:


a != c


# Because we assigned `c` as a function of `a`,  `c` and `a` have the same values but reference different objects in memory.

# ***Table 2-3*** in the textbook summarizes the binary operators.
# 
# - `a + b` : Add a and b
# - `a - b` : Subtract b from a
# - `a * b` : Multiply a by b
# - `a / b` : Divide a by b
# - `a // b` : Floor-divide a by b, dropping any fractional remainder
# - `a ** b` : Raise a to the b power
# - `a & b` : True if both a and b are True; for integers, take the bitwise AND
# - `a | b` : True if either a or b is True; for integers, take the bitwise OR
# - `a ^ b` : For booleans, True if a or b is True , but not both; for integers, take the bitwise EXCLUSIVE-OR
# - `a == b` : True if a equals b
# - `a != b`: True if a is not equal to b
# - `a <= b, a < b` : True if a is less than (less than or equal) to b
# - `a > b, a >= b`: True if a is greater than (greater than or equal) to b
# - `a is b` : True if a and b reference the same Python object
# - `a is not b` : True if a and b reference different Python objects

# ### Mutable and immutable objects
# 
# > Most objects in Python, such as lists, dicts, NumPy arrays, and most user-defined
# types (classes), are mutable. This means that the object or values that they contain can
# be modified.
# 
# Lists are mutable, therefore we can modify them.

# In[49]:


a_list = ['foo', 2, [4, 5]]


# ***Python is zero-indexed! The first element has a zero subscript `[0]`!***

# In[50]:


a_list[0]


# In[51]:


a_list[2]


# In[52]:


a_list[2] = (3, 4)


# Tuples are *immutable*, therefore we cannot modify them.

# In[53]:


a_tuple = (3, 5, (4, 5))


# The Python interpreter returns an error is we try to modify `a_tuple` becuase tuples are immutable.

# In[54]:


# a_tuple[1] = 'four'


# ## Scalar Types
# 
# > Python along with its standard library has a small set of built-in types for handling numerical data, strings, boolean ( True or False ) values, and dates and time. These "single value" types are sometimes called scalar types and we refer to them in this book as scalars. See Table 2-4 for a list of the main scalar types. Date and time handling will be discussed separately, as these are provided by the datetime module in the standard  library.
# 
# ***Table 2-4*** from the textbook lists the most common scalar types.
# 
# - `None`: The Python "null" value (only one instance of the None object exists)
# - `str`: String type; holds Unicode (UTF-8 encoded) strings
# - `bytes`: Raw ASCII bytes (or Unicode encoded as bytes)
# - `float`: Double-precision (64-bit) floating-point number (note there is no separate double type)
# - `bool`: A True or False value
# - `int`: Arbitrary precision signed integer

# ### Numeric types
# 
# In Python, integers are unbounded, and `**` raises numbers to a power.
# So, `ival ** 6` is $17239781^6$.

# In[55]:


ival = 17239871
ival ** 6


# Floats (decimal numbers) are 64-bit in Python.

# In[56]:


fval = 7.243


# In[57]:


type(fval)


# Dividing integers yields a float, if necessary.

# In[58]:


3 / 2


# If we want C-style integer division (i.e., $3 / 2 = 1$), we have to use `//` (i.e., floor division).

# In[59]:


3 // 2


# ### Strings
# 
# We will skip strings for a first course.

# ### Booleans
# 
# > The two boolean values in Python are written as True and False . Comparisons and other conditional expressions evaluate to either True or False . Boolean values are combined with the and and or keywords.
# 
# Note the capitalization, which may be different from your prior programming languages.

# In[60]:


True and True


# In[61]:


False or True


# We can also substitute `&` for `and` and `|` for `or`.

# In[62]:


True & True


# In[63]:


False | True


# ### Type casting
# 
# We can "recast" variables to change their types.

# In[64]:


s = '3.14159'


# In[65]:


# 1 + s


# In[66]:


1 + float(s)


# In[67]:


fval = float(s)


# In[68]:


type(fval)


# In[69]:


int(fval)


# In[70]:


bool(fval)


# In[71]:


bool(0)


# Anything other than zero converts to a boolean `True`.

# In[72]:


bool(-1)


# We recast the string `'5'` to an integer or the integer `5` to a string to prevent the `5 + '5'` error above.

# In[73]:


5 + int('5')


# In[74]:


str(5) + '5'


# ### None

# In Python, `None` is null.
# `None` is like `#N/A` or `=na()` in Excel.

# In[75]:


a = None
a is None


# In[76]:


b = 5
b is not None


# In[77]:


type(None)


# ## Control Flow
# 
# > Python has several built-in keywords for conditional logic, loops, and other standard control flow concepts found in other programming languages.
# 
# If you understand Excel's `if()`, then you understand Python's `if`, `elif`, and `else`.

# ### if, elif, and else
# 
# In general, I prefer single quotes because they do not require pressing the shift key.
# However, in the following example, we must use double quotes to avoid confusion with the apostrophe in "it's".

# In[78]:


x = -1


# In[79]:


type(x)


# In[80]:


if x < 0:
    print("It's negative")


# Python's `elif` avoids Excel's nested `if()`s.
# `elif` continues an `if` block, and `else` runs if the other conditions are not met.

# In[81]:


x = 10
if x < 0:
    print("It's negative")
elif x == 0:
    print('Equal to zero')
elif 0 < x < 5:
    print('Positive but smaller than 5')
else:
    print('Positive and larger than or equal to 5')


# We can combine comparisons with `and` and `or`.

# In[82]:


a = 5
b = 7
c = 8
d = 4
if a < b or c > d:
    print('Made it')


# ### for loops
# 
# We use `for` loops to loop over a collection, like a list or tuple.
# 
# Note that the following examples assign values with `+=`.
# `a += 5` is an abbreviation for `a = a + 5`.
# There are similar abbreviations for subtraction, multiplication, and division, too (`-=`, `*=`, and `/=`).
# 
# The `continue` keyword skips the remainder of the `if` block for that loop iteration.

# In[83]:


sequence = [1, 2, None, 4, None, 5]
total = 0
for value in sequence:
    if value is None:
        continue
    total += value


# The `break` keyword exits the loop altogether.

# In[84]:


sequence = [1, 2, 0, 4, 6, 5, 2, 1]
total_until_5 = 0
for value in sequence:
    if value == 5:
        break
    total_until_5 += value


# ### range
# 
# > The range function returns an iterator that yields a sequence of evenly spaced
# integers.
# 
# - With one argument, `range()` creates an iterator from 0 to that number *but excluding that number* (so `range(10)` as an interator with a length of 10 that starts at 0)
# - With two arguments, the first argument is the *inclusive* starting value, and the second argument is the *exclusive* ending value
# - With three arguments, the third argument is the iterator step size
# 
# One advantage of `range()` is that is memory efficient because it does not expand to a list.

# In[85]:


range(10)


# In[86]:


list(range(10))


# In[87]:


list(range(1, 10))


# In[88]:


list(range(1, 10, 1))


# In[89]:


list(range(0, 20, 2))


# In Python, intervals are "closed" (inclusive or square bracket) on the left and "open" (exclusive or parentheses) on the right.
# The following is an empty list because we cannot count from 5 to 0 by steps of +1.

# In[90]:


list(range(5, 0))


# In[91]:


list(range(5, 0, -1))


# In[92]:


seq = [1, 2, 3, 4]
for i in range(len(seq)):
    val = seq[i]


# We can loop over the list `seq` directly.
# The following code is equivalent to the previous code, but is more "Pythonic".

# In[93]:


for i in seq:
    val = i


# The modulo operator (`%`) returns the remainder.
# For example, two goes into five twice with a remainder of one.

# In[94]:


5 // 2


# In[95]:


5 % 2


# In[96]:


sum = 0
for i in range(100000):
    if i % 3 == 0 or i % 5 == 0:
        sum += i


# ### Ternary expressions
# 
# We said above that Python `if` and `else` is cumbersome relative to Excel's `if()`.
# We can complete simple comparisons on one line in Python.

# In[97]:


x = 5
value = 'Non-negative' if x >= 0 else 'Negative'


# ## Practice

# ***Practice:*** 
# Extract the year, month, and day from an 8-digit date (i.e., YYYYMMDD format).
# Try `20080915` using `//` (integer division) and `%` (modulo division, which returns the remainder).

# In[98]:


5 / 2


# In[99]:


5 // 2


# In[100]:


5 % 2


# In[101]:


ymd = 20080915
year = ymd // 10000
year


# ***Practice:***
# Convert your code above in to a function `date` that parses 8-digit integer dates and returns a tuple the the year, month, and date (i.e., `return (year, month, date)`)

# ***Practice:***
# Write a for loop that prints the squares of integers from 1 to 10.

# ***Practice:*** 
# Write a for loop that sums the squares of integers from 1 to 10 while the sum is less than 50.

# ***Practice:*** 
# Write a for loop that sums the squares of ***even*** integers from 1 to 10 while the sum is less than 50.

# ***Practice:*** 
# Write a for loop that prints the numbers from 1 to 100. 
# For multiples of three print "Fizz" instead of the number.
# For multiples of five print "Buzz". 
# For numbers that are multiples of both three and five print "FizzBuzz".
# More [here](https://blog.codinghorror.com/why-cant-programmers-program/).

# ***Practice:*** 
# Use ternary expressions to make your FizzBuzz code more compact.

# ***Practice:***
# Given a positive integer $N$, print a numerical triangle of height $N-1$ like the one below for $N=6$.
# 
# ```
# 1
# 22
# 333
# 4444
# 55555
# ```

# You can find many more Python coding challenges at <www.hackerrank.com>.
# These challenges are fun but not necessary for our course because we will spend our time using higher-level features.
