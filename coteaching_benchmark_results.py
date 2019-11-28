
# coding: utf-8

# ## Use this to read in the scores for co-teaching. take the max score of both models for the fairest comparison.

# In[1]:


import os
import numpy as np


# In[2]:


def read_last_line(filename):
    with open(filename, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR) 
        last_line = f.readline().decode()
    return last_line


# In[4]:


basedir = '/home/cgn/coteaching_results/'
for f in [f for f in os.listdir(basedir) if '_' in f]:
    print(f, end=': ')
    result = read_last_line(basedir + f +"/out_{}.log".format(f))
    model1_score = float(result.split('Model1')[-1][:8])
    model2_score = float(result.split('Model2')[-1][:8])
    score = max(model1_score, model2_score)
    print(score)

