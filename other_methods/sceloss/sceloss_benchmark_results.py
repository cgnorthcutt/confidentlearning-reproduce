
# coding: utf-8

# ## Use this to read in the scores for SCELoss. take the max score of both models for the fairest comparison.

# In[1]:


import os
import numpy as np


# In[3]:


def get_scores(filename):
    with open(filename, 'r') as f:
        results = f.readlines()[-6:-2]
    acc1 = float(results[0].split("\t")[-1].strip())
    acc1best = float(results[1].split("\t")[-1].strip())
    acc5 = float(results[2].split("\t")[-1].strip())
    acc5best = float(results[3].split("\t")[-1].strip())
    return {
        'acc1': acc1,
        'acc1best': acc1best,
        'acc5': acc5,
        'acc5best': acc5best,
    }


# In[4]:


basedir = '/home/cgn/sceloss_results/'
for f in [f for f in os.listdir(basedir) if '_' in f]:
    print(f, end=': ')
    result = get_scores(basedir + f +"/out_{}.log".format(f))
    print(result['acc1'])
#     model1_score = float(result.split('Model1')[-1][:8])
#     model2_score = float(result.split('Model2')[-1][:8])
#     score = max(model1_score, model2_score)
#     print(score)


# In[22]:


basedir = '/home/cgn/sceloss_results/'
for f in [f for f in os.listdir(basedir) if '_' in f]:
    print(f, end=': ')
    result = get_scores(basedir + f +"/out_{}.log".format(f))
    print(result['acc1'])
#     model1_score = float(result.split('Model1')[-1][:8])
#     model2_score = float(result.split('Model2')[-1][:8])
#     score = max(model1_score, model2_score)
#     print(score)

