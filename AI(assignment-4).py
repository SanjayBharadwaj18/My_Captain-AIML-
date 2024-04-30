#!/usr/bin/env python
# coding: utf-8

# In[4]:


E = {0, 2, 4, 6, 8}
N = {1, 2, 3, 4, 5}


# In[5]:


union_set = E.union(N)
intersection_set = E.intersection(N)
difference_set = E.difference(N)
symmetric_difference_set = E.symmetric_difference(N)


# In[6]:


print("Union of E and N is", union_set)
print("Intersection of E and N is", intersection_set)
print("Difference of E and N is", difference_set)
print("Symmetric difference of E and N is", symmetric_difference_set)


# In[ ]:




