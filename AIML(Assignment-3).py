#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence

# Example usage:
n = 20  # Change n to generate Fibonacci sequence up to a different number
fibonacci_sequence = fibonacci(n)
print("Fibonacci sequence up to", n, ":", fibonacci_sequence)


# In[ ]:




