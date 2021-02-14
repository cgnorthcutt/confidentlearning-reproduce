#!/usr/bin/env python
# coding: utf-8

# # All CIFAR experiments in the Confident Learning paper.
# 
# ### Benchmark test@1 accuracies for non-cl methods in the were implemented interally at Google.
# ### Use this to read in the scores for Confident Learning, with the final
# ### training using co-teaching. take max score of both models for fair comparison.

# In[1]:


import numpy as np
import cleanlab
from cleanlab import baseline_methods
from cleanlab.latent_estimation import compute_confident_joint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
import sys
import json

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
font = {'family': 'sans-serif',
        'sans-serif':['Helvetica'],
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)


# # Confident Learning ACC/PRECISION/RECALL/F1 at finding label errors in CIFAR-10

# In[9]:


noisy_base_dir = '/home/cgn/cgn/confidentlearning-reproduce/cifar10/'
base_dir = noisy_base_dir # '/home/cgn/OLD_BAD_cifar10/'


# In[10]:


rfn = '/datasets/datasets/cifar10/cifar10/train_filename2label.json'
with open(rfn, 'r') as rf:
    d = json.load(rf)
y = np.asarray([v for k,v in d.items()])


# In[11]:


# This code assumes that the `cifar10` folder contains trained models.

folders = [c for c in os.listdir(base_dir) if '__' in c]
results = []
for folder in sorted([f for f in folders if f != '__pycache__']):
    print(folder)
    psx_file = [z for z in os.listdir(base_dir + folder) if 'pyx' in z][0]
    print(psx_file)
    psx = np.load(base_dir + folder + "/" + psx_file)
    
    #Make sure psx is the right shape
    psx = psx[:,:10]
    
    # Load noisy labels
    frac_zero_noise_rates = folder.split('_')[-7]
    noise_amount = folder.split('_')[-1]
    if noise_amount == '8':
        continue
    rfn = 'cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json'.format(
        frac_zero_noise_rates, noise_amount)
    with open(noisy_base_dir + "cifar10_noisy_labels/" + rfn, 'r') as rf:
        d = json.load(rf)
    s = np.asarray([v for k,v in d.items()])
    
    true_label_errors = s != y
    acc = np.sum(s == y) / len(y)
    print('accuracy of labels:', acc)

    # Benchmarks
    
    label_error_mask = np.zeros(len(s), dtype=bool)
    label_error_indices = compute_confident_joint(
        s, psx, return_indices_of_off_diagonals=True
    )[1]
    for idx in label_error_indices:
        label_error_mask[idx] = True
    conf_joint_only = label_error_mask

#     # Confident learning optimized
#     best_f1 = -1
#     cl_opt = None
#     for prune_method in ['prune_by_class', 'prune_by_noise_rate', 'both']:
#         label_errs = cleanlab.pruning.get_noise_indices(
#             s,
#             psx,
#             prune_method=prune_method,
#         )
#         f1 = precision_recall_fscore_support(
#             y_true=true_label_errors,
#             y_pred=label_errs,
#         )[2][0]

#         if f1 > best_f1:
#             print(prune_method)
#             best_f1 = f1
#             cl_opt = label_errs

    results.append({
        'noise_amount_acc': acc,
        'noise_amount': noise_amount,
        'frac_zero_noise_rates': frac_zero_noise_rates,
        'argmax' : confusion_matrix(
            y_true=true_label_errors,
            y_pred=baseline_methods.baseline_argmax(psx, s),
        ),
        'argmax_cm': confusion_matrix(
            y_true=true_label_errors,
            y_pred=baseline_methods.baseline_argmax_confusion_matrix(psx, s),
        ),
        'argmax_ccm': confusion_matrix(
            y_true=true_label_errors,
            y_pred=baseline_methods.baseline_argmax_calibrated_confusion_matrix(
                psx, s),
        ),
        'conf_joint_only': confusion_matrix(
            y_true=true_label_errors,
            y_pred=conf_joint_only,
        ),
        'cl_pbnr': confusion_matrix(
            y_true=true_label_errors,
            y_pred=cleanlab.pruning.get_noise_indices(s, psx),
        ),
        'cl_pbc': confusion_matrix(
            y_true=true_label_errors,
            y_pred=cleanlab.pruning.get_noise_indices(
                s, psx, prune_method='prune_by_class'),
        ),
        'cl_both': confusion_matrix(
            y_true=true_label_errors,
            y_pred=cleanlab.pruning.get_noise_indices(
                s, psx, prune_method='both'),
        ),
#         'cl_opt': confusion_matrix(
#             y_true=true_label_errors,
#             y_pred=label_errs,
#         ),
    })
    
    print()


# In[12]:


methods = ['argmax', 'argmax_ccm', 'argmax_cm', 'cl_pbnr', 'cl_pbc', 'cl_both', 'conf_joint_only']
precision_func = lambda x: x[1][1] * 1.0 / (x[1][1] + x[0][1])
recall_func = lambda x: x[1][1] * 1.0 / (x[1][1] + x[1][0])
acc_func = lambda x: (x[1][1] + x[0][0]) * 1.0 / (x[1][1] + x[1][0] + x[0][1] + x[0][0])
f1_func = lambda x: 2 * precision_func(x) * recall_func(x) /     (precision_func(x) + recall_func(x))
scoring = {
    'precision': precision_func,
    'recall': recall_func,
    'f1': f1_func,
    'acc': acc_func
}

# Create pandas dataframe to view results
df = pd.DataFrame(results)
df['label_error_fraction'] = np.round(1 - df['noise_amount_acc'], 1)

# For each baseline method, apply each scoring method
for method in methods:
    for k, v in scoring.items():
        df[method + "_" + k] = df[method].apply(v)


# In[13]:


if type(df['frac_zero_noise_rates'].iloc[0]) is str:
    df['frac_zero_noise_rates'] = df['frac_zero_noise_rates'].apply(
        lambda x: int(x) / 10.)
if type(df['noise_amount'].iloc[0]) is str:
    df['noise_amount'] = df['noise_amount'].apply(
        lambda x: int(x) / 10.)


# In[14]:


a = df[(df['frac_zero_noise_rates']!=.4) & (df['frac_zero_noise_rates']!=.2) & (df['label_error_fraction'] != 0)].groupby(['label_error_fraction', 'frac_zero_noise_rates']).mean()[['argmax_acc', 'cl_pbc_acc', 'cl_pbnr_acc', 'cl_both_acc', 'conf_joint_only_acc']].T.round(2)
a


# In[15]:


b = df[(df['frac_zero_noise_rates']!=.4) & (df['frac_zero_noise_rates']!=.2) & (df['label_error_fraction'] != 0)].groupby(['label_error_fraction', 'frac_zero_noise_rates']).mean()[['argmax_f1', 'cl_pbc_f1', 'cl_pbnr_f1', 'cl_both_f1', 'conf_joint_only_f1']].T.round(2)
b


# In[16]:


c = df[(df['frac_zero_noise_rates']!=.4) & (df['frac_zero_noise_rates']!=.2) & (df['label_error_fraction'] != 0)].groupby(['label_error_fraction', 'frac_zero_noise_rates']).mean()[['argmax_precision', 'cl_pbc_precision', 'cl_pbnr_precision', 'cl_both_precision', 'conf_joint_only_precision']].T.round(2)
c


# In[17]:


d = df[(df['frac_zero_noise_rates']!=.4) & (df['frac_zero_noise_rates']!=.2) & (df['label_error_fraction'] != 0)].groupby(['label_error_fraction', 'frac_zero_noise_rates']).mean()[['argmax_recall', 'cl_pbc_recall', 'cl_pbnr_recall', 'cl_both_recall', 'conf_joint_only_recall']].T.round(2)
d


# In[109]:


for z in [a,b,c,d]:
    print(z.to_latex())


# # Set up training experiments

# In[18]:


folders = [c for c in os.listdir(base_dir) if 'noise_amount' in c]
results = []
for folder in sorted(folders):
    print(folder)
    psx_file = [z for z in os.listdir(base_dir + folder) if 'pyx' in z][0]
    psx = np.load(base_dir + folder + "/" + psx_file)
    
    #Make sure psx is the right shape
    psx = psx[:,:10]
    
    # Load noisy labels
    frac_zero_noise_rates = folder.split('_')[-7]
    noise_amount = folder.split('_')[-1]
    rfn = 'cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json'.format(
        frac_zero_noise_rates, noise_amount)
    with open(noisy_base_dir + "cifar10_noisy_labels/" + rfn, 'r') as rf:
        d = json.load(rf)
    s = np.asarray([v for k,v in d.items()])
    
    true_label_errors = s != y
    acc = np.sum(s == y) / len(y)
    print('accuracy of labels:', acc)

    # Benchmarks
    
    label_error_mask = np.zeros(len(s), dtype=bool)
    label_error_indices = compute_confident_joint(
        s, psx, return_indices_of_off_diagonals=True
    )[1]
    for idx in label_error_indices:
        label_error_mask[idx] = True
    baseline_conf_joint_only = label_error_mask
    
    baseline_argmax = baseline_methods.baseline_argmax(psx, s)
    
    baseline_cl_pbc = cleanlab.pruning.get_noise_indices(
                s, psx, prune_method='prune_by_class')
    
    baseline_cl_pbnr = cleanlab.pruning.get_noise_indices(
                s, psx, prune_method='prune_by_noise_rate')
    
    baseline_cl_both = cleanlab.pruning.get_noise_indices(
                s, psx, prune_method='both')

    # Create folders for and store masks for training.
    new_folder = base_dir + folder + "/train_pruned_conf_joint_only/"
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass
    np.save(new_folder + "train_mask.npy", ~baseline_conf_joint_only)
    
    new_folder = base_dir + folder + "/train_pruned_argmax/"
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass
    np.save(new_folder + "train_mask.npy", ~baseline_argmax)
    
    new_folder = base_dir + folder + "/train_pruned_cl_pbc/"
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass
    np.save(new_folder + "train_mask.npy", ~baseline_cl_pbc)
    
    new_folder = base_dir + folder + "/train_pruned_cl_pbnr/"
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass
    np.save(new_folder + "train_mask.npy", ~baseline_cl_pbnr)
    
    new_folder = base_dir + folder + "/train_pruned_cl_both/"
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass
    np.save(new_folder + "train_mask.npy", ~baseline_cl_both)
    print()


# ## Estimate noise (used as forget_rate in Co-Teaching) based on cross-val accuracy 

# In[19]:


import numpy as np
from sklearn.metrics import accuracy_score


# In[20]:


print("Est Noise based on cross-val predictions:")
base = "/home/cgn/cifar10_psx_crossval/"
for f in sorted(sorted(os.listdir(base)), key=lambda x: x.split("_")[-1]):
    fn = base + f + "/" + [z for z in os.listdir(base + f) if 'pyx' in z][0]
    psx = np.load(fn)
    pred = psx.argmax(axis=1)
    sparsity, noise = f.split("_")
    noise = '6' if noise == '7' else noise
    with open('/datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json'.format(sparsity, noise), 'r') as rf:
        labels = json.load(rf)
    print(f, "| Noise:",  round(1 - accuracy_score(list(labels.values()), pred), 2))


# # Analyze accuracy of finding label errors

# In[21]:


import json
base="/home/cgn/cgn/confidentlearning-reproduce/cifar10/"
for sparsity in [0,2,4,6]:
    for noise in [2, 4, 6]:
        with open(base + "cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json".format(sparsity, noise), 'r') as rf:
            d = json.load(rf)
            acc = sum(list(d.values()) == y) / len(y)
            print('sparsity:', sparsity, 'noise:', noise, 'noisy labels acc:', acc)
            assert 10 - (round(acc * 10)) == (noise if noise < 6 else 7)


# In[22]:


import json
stuff = []
base="/home/cgn/cgn/confidentlearning-reproduce/cifar10/"
for sparsity in [0,2,4,6]:
    for noise in [2, 4, 6]:
        with open(base + "cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json".format(sparsity, noise), 'r') as rf:
            s = list(json.load(rf).values())
        for method in ['cl_both', 'cl_pbnr', 'cl_pbc', 'conf_joint_only', 'argmax']:
            with open(base + "cifar10_noisy_labels__frac_zero_noise_rates__0_{}__noise_amount__0_{}/train_pruned_{}/train_mask.npy".format(sparsity, noise, method), 'rb') as rf:
                train_mask = np.load(rf)  
            acc = sum(s == y) / len(y)
#             print('sparsity:', sparsity, 'noise:', noise, 'method', method, '\t\tnoisy labels acc:', acc)
#             assert 10 - (round(acc * 10)) == (noise if noise < 6 else 7)
            stuff.append({
                'sparsity': sparsity,
                'noise': noise,
                'method': method,
                'acc(s,y)': acc,
                'acc_pruned(s,y)': sum(np.asarray(s)[train_mask] == y[train_mask]) / sum(train_mask),
                'acc_mask': sum((s == y) == train_mask) / len(y),
            })
            


# In[23]:


base = '/home/cgn/masked_imagenet_training/resnet18/'
for removed in [20, 40, 60, 80, 100]:
    path = base + 'resnet18_{}_cj_only'.format(removed) + "/" + "out.log"
    try:
        with open(path, 'r') as rf:
            lines = rf.readlines()
    except:
        continue
    acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
    plt.figure(figsize=(10,5))
    plt.plot(acc)
    plt.show()
    print(path + "\n\n", flush=True)


# In[ ]:


base = '/home/cgn/masked_imagenet_coteaching/resnet18/cj_only/'
for removed in [20, 40, 60, 80, 100]:
    for fr in [1, 2, 3, 4]:
            path = base + str(removed) + "/" + str(fr) + "/" + "out.log"
            try:
                with open(path, 'r') as rf:
                    lines = rf.readlines()
            except:
                continue
            acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
            model1, model2 = acc[::2], acc[1::2]
            acc1 = [max(model2[i][0], a[0]) for i, a in enumerate(model1)]
            acc5 = [max(model2[i][1], a[1]) for i, a in enumerate(model1)]
            plt.figure(figsize=(10,5))
            plt.plot(acc1, )
            plt.show()
            print(path + "\n\n", flush=True)


# In[ ]:


base = '/home/cgn/cifar10_cl_coteaching/final/'
for noise in [2, 4, 6]:
    for experiment in ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']:
        for sparsity in [0, 2, 4, 6]:
            f = str(sparsity) + "_" + str(noise)
            path = base + experiment + "/" + f + "/out.log"
            with open(path, 'r') as rf:
                lines = rf.readlines()
            acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
            model1, model2 = acc[::2], acc[1::2]
            acc1 = [max(model2[i][0], a[0]) for i, a in enumerate(model1)]
            acc5 = [max(model2[i][1], a[1]) for i, a in enumerate(model1)]
            plt.figure(figsize=(10,5))
            plt.plot(acc1, )
            plt.show()
            print(experiment + " " + str(sparsity) + "_" + str(noise) + " | noise: " + str(noise) + " | sparsity: " + str(sparsity) + "\n\n", flush=True)


# In[ ]:


base = '/home/cgn/cifar10_cl_coteaching/'
for fr in [1]:  # , 2, 3, 4, 5
    for noise in [2, 4, 6]:
        for experiment in ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']:
            for sparsity in [0, 2, 4, 6]:
                f = str(sparsity) + "_" + str(noise)
                path = base + "fr" + str(fr) + "/" + experiment + "/" + f + "/out_" + f + ".log"
                with open(path, 'r') as rf:
                    lines = rf.readlines()
                acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
                model1, model2 = acc[::2], acc[1::2]
                acc1 = [max(model2[i][0], a[0]) for i, a in enumerate(model1)]
                acc5 = [max(model2[i][1], a[1]) for i, a in enumerate(model1)]
                plt.figure(figsize=(10,5))
                plt.plot(acc1, )
                plt.show()
                print(experiment + " " + str(sparsity) + "_" + str(noise) + " | noise: " + str(noise) + " | sparsity: " + str(sparsity) + "\n\n", flush=True)


# In[24]:


dfs = []
for i, sdf in pd.DataFrame(stuff).groupby('method'):
    dfs.append(pd.DataFrame(sdf.sort_values(by=['noise', 'sparsity']).set_index(
            ['noise', 'sparsity'])['acc(s,y)']).T.set_index([[i]]))
df_results = pd.concat(dfs).round(2)
print('Accuracy of noisy labels compared to ground truth labels. (should match noise)')
df_results


# In[25]:


dfs = []
for i, sdf in pd.DataFrame(stuff).groupby('method'):
    dfs.append(pd.DataFrame(sdf.sort_values(by=['noise', 'sparsity']).set_index(
            ['noise', 'sparsity'])['acc_pruned(s,y)']).T.set_index([[i]]))
df_results = pd.concat(dfs).round(2)
print("Accuracy of cleaned labels after CL has removed noisy data.")
df_results


# In[26]:


dfs = []
for i, sdf in pd.DataFrame(stuff).groupby('method'):
    dfs.append(pd.DataFrame(sdf.sort_values(by=['noise', 'sparsity']).set_index(
            ['noise', 'sparsity'])['acc_mask']).T.set_index([[i]]))
df_results = pd.concat(dfs).round(2)
print("Accuracy finding label errors.")
df_results


# # Benchmarking Confident Learning (final training with CO-TRAINING):
# ## learning with noisy labels accuracy 

# In[9]:


import subprocess
import pandas as pd
import numpy as np


# In[10]:


base = '/home/cgn/cifar10_cl_coteaching/'


# In[11]:


experiments = ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']


# In[12]:


dfs = {}
for batch_size in [128, 256, 512]:
    results = []
    for experiment in ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']:
        for noise in [2, 4, 6]:
            for sparsity in [0, 2, 4, 6]:
                f = str(sparsity) + "_" + str(noise)
                path = base + "final_{}/".format(batch_size) + experiment + "/" + f + "/out.log"
                try:
                    with open(path, 'r') as rf:
                        lines = rf.readlines()
                    acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
                    model1, model2 = acc[::2], acc[1::2]
                    acc1 = max([max(model2[i][0], a[0]) for i, a in enumerate(model1)])
                    acc5 = max([max(model2[i][1], a[1]) for i, a in enumerate(model1)])
                except:
                    acc1, acc5 = np.NaN, np.NaN
                results.append({
                    'experiment': experiment,
                    'frac_zero_noise_rates': sparsity / 10.,
                    'noise_amount': noise / 10.,
                    'acc1': acc1,
                    'acc5': acc5,
                    'forget_rate': noise / 10. / 2,
                })

    df = pd.DataFrame(results)
    df_results = pd.concat([
        z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
            ['noise_amount', 'frac_zero_noise_rates']).drop(
            ['experiment', 'forget_rate', 'acc5'], axis=1).T.set_index([[i]]) \
        for i, z in df.groupby('experiment')
    ])
    dfs[batch_size] = df_results
    
cl = pd.concat(dfs.values()).max(level=0)


# In[58]:


# Test out the effect of different forget rates

# results = []
# for fr in [1, 2, 3, 4, 5]:
#     for experiment in ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']:
#         for noise in [2, 4, 6]:
#             for sparsity in [0, 2, 4, 6]:
#                 f = str(sparsity) + "_" + str(noise)
#                 path = base + "fr" + str(fr) + "/" + experiment + "/" + f + "/out_" + f + ".log"
#                 with open(path, 'r') as rf:
#                     lines = rf.readlines()
# #                 acc = [l for l in lines if l.startswith('Epoch [250/250] Test')][0].split("Model")[1:]
# #                 acc = max([float(a[2:9]) for a in acc])
#                 try:
#                     with open(path, 'r') as rf:
#                         lines = rf.readlines()
#                     acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
#                     model1, model2 = acc[::2], acc[1::2]
#                     acc1 = max([max(model2[i][0], a[0]) for i, a in enumerate(model1)])
#                     acc5 = max([max(model2[i][1], a[1]) for i, a in enumerate(model1)])
#                 except:
#                     acc1, acc5 = np.NaN, np.NaN
#                 results.append({
#                     'experiment': experiment,
#                     'frac_zero_noise_rates': sparsity / 10.,
#                     'noise_amount': noise / 10.,
#                     'acc': acc1,
#                     'forget_rate': fr,
#                 })

# df_results = pd.concat([
#     z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
#         ['noise_amount', 'frac_zero_noise_rates']).drop(
#         ['experiment', 'forget_rate'], axis=1).T.set_index([[i]]) \
#     for i, z in pd.DataFrame(results).groupby(['experiment', 'forget_rate'])  # ['forget_rate', 'experiment']
# ])
# display(df_results)

# See which forget rate is maximal for each setting

# lol = pd.DataFrame(results).groupby(
#     ['frac_zero_noise_rates', 'experiment', 'noise_amount']).apply(
#     lambda z: z['forget_rate'][z['acc'].idxmax()]).reset_index()
# df_results = pd.concat([
#     z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
#         ['noise_amount', 'frac_zero_noise_rates']).drop(
#         ['experiment'], axis=1).T.set_index([[i]]) \
#     for i, z in lol.groupby('experiment')
# ])
# display(df_results)

# Almost the final results, but this does not consider the best batch size.

# df = pd.DataFrame(results)
# df = df[df['forget_rate'] == df['noise_amount'] * 10 // 2].drop('forget_rate', axis=1)
# df_results = pd.concat([
#     z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
#         ['noise_amount', 'frac_zero_noise_rates']).drop(
#         ['experiment'], axis=1).T.set_index([[i]]) \
#     for i, z in df.groupby('experiment')
# ])
# almost_final = (df_results / 100).round(4)
# display(almost_final)


# In[13]:


# Find and save the batch size used at each setting.
df_batch_size = (cl == dfs[128]).astype(int) * 128 + (cl == dfs[256]).astype(int) * 128 + (cl == dfs[512]).astype(int) * 512
df_batch_size[df_batch_size > 512] = 512
df_batch_size[df_batch_size == 512] = 450  # 450 batch size was used (no reason)
pd.to_pickle(df_batch_size, 'batch_sizes_df.p')


# In[66]:


display(dfs[128])
display(dfs[256])
display(dfs[512])
display(cl)


# In[143]:


lol = pd.DataFrame(results).groupby(['frac_zero_noise_rates', 'experiment', 'noise_amount'])['acc'].max().reset_index()
df_results = pd.concat([
    z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
        ['noise_amount', 'frac_zero_noise_rates']).drop(
        ['experiment'], axis=1).T.set_index([[i]]) \
    for i, z in lol.groupby('experiment')
])
(df_results / 100).round(3)


# In[230]:


cl


# In[231]:


# df_results = pd.concat([
#     z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
#         ['noise_amount', 'frac_zero_noise_rates']).drop(
#         ['experiment'], axis=1).T.set_index([['OURS: ' + i]]) \
#     for i, z in pd.DataFrame(results).groupby('experiment')
# ])

## Results on other models using google code by Lu Jiang (author of MentorNet)

mentornet = [[
#     0.9378,  # 0 noise
    0.8493, 0.8514, 0.8319, 0.8342,  # 0.2 noise
    0.6444, 0.6423, 0.6238, 0.6146,  # 0.4 noise
    0.2996, 0.3160, 0.2930, 0.2786,  # 0.6 noise
]]
mentornet = pd.DataFrame(mentornet, columns=cl.columns, index = ['mentornet'])
smodel = [[
#     0.9375,  # 0 noise
    0.8000, 0.7996, 0.7974, 0.7910,  # 0.2 noise
    0.5856, 0.6121, 0.5913, 0.5752,  # 0.4 noise
    0.2845, 0.2853, 0.2793, 0.2726,  # 0.6 noise
]]
smodel = pd.DataFrame(smodel, columns=cl.columns, index = ['smodel'])
reed = [[
#     0.9372,  # 0 noise
    0.7809, 0.7892, 0.8076, 0.7927,  # 0.2 noise
    0.6048, 0.6041, 0.6124, 0.5860,  # 0.4 noise
    0.2904, 0.2939, 0.2913, 0.2677,  # 0.6 noise
]]
reed = pd.DataFrame(reed, columns=cl.columns, index = ['reed'])
vanilla = [[
#     0.935,  # 0 noise
    0.7843, 0.7916, 0.7901, 0.7825,  # 0.2 noise
    0.6022, 0.6077, 0.5963, 0.5727,  # 0.4 noise
    0.2696, 0.2966, 0.2824, 0.2681,  # 0.6 noise
]]
vanilla = pd.DataFrame(vanilla, columns=cl.columns, index = ['vanilla'])


# In[235]:


# Final table of results.
cifar10_final_benchmarks = (cl / 100).append(mentornet).append(smodel).append(reed).append(vanilla) * 100
# Add average column
cifar10_final_benchmarks.insert(0, 'AVG', cifar10_final_benchmarks.mean(numeric_only=True, axis=1))
# cifar10_final_benchmarks.to_csv('cifar10/benchmarks.csv')
cifar10_final_benchmarks.round(1)


# In[236]:


# Latex of the final table in the paper
print(cifar10_final_benchmarks.round(1).to_latex())


# # Benchmarking Confident Learning (final training with Vanilla training):
# ## learning with noisy labels accuracy 

# In[42]:


import subprocess


# In[43]:


base = '/home/cgn/cgn/cleanlab/examples/cifar/cifar10/'


# In[44]:


experiments = ['train_pruned_argmax', 'train_pruned_cl_pbc', 'train_pruned_cl_pbnr', 'train_pruned_cl_both', 'train_pruned_conf_joint_only']


# In[45]:


results = []
for settings in sorted([f for f in os.listdir(base) if 'noise' in f]):
    for experiment in experiments:
        frac_zero_noise_rates = settings.split('_')[-7]
        noise_amount = settings.split('_')[-1]
        # Remove results with noise fraction 0.8 (way too high for any practical case)
        if noise_amount != '8':
            try:
                cmd = 'python3 {}cifar10_train_crossval.py /datasets/datasets/cifar10/cifar10/ --resume {} --evaluate --gpu 0'.format(
                base, base + settings + "/" + experiment + "/" + 'model_resnet50__masked_best.pth.tar')
                result = subprocess.check_output(cmd, shell=True)
            except:
                cmd = 'python3 {}cifar10_train_crossval2.py /datasets/datasets/cifar10/cifar10/ --resume {} --evaluate --gpu 0'.format(
                base, base + settings + "/" + experiment + "/" + 'model_resnet50__masked_best.pth.tar')
                result = subprocess.check_output(cmd, shell=True)
            acc1, _, acc5 = result.split(b"* Acc@1 ")[-1].strip().split()
            acc1, acc5 = float(acc1), float(acc5)
            results.append({
                'experiment': experiment[13:],
                'frac_zero_noise_rates': frac_zero_noise_rates,
                'noise_amount': noise_amount,
                'acc1': acc1,
                'acc5': acc5,
            })
            print(results[-1])


# In[46]:


df_results = pd.concat([
    z.sort_values(by=['noise_amount', 'frac_zero_noise_rates']).set_index(
        ['noise_amount', 'frac_zero_noise_rates']).drop(
        ['acc5', 'experiment'], axis=1).T.set_index([['OURS: ' + i]]) \
    for i, z in pd.DataFrame(results).groupby('experiment')
])

## Results on other models using google code by Lu Jiang (author of MentorNet)

mentornet = [[
    0.9378,  # 0 noise
    0.8493, 0.8514, 0.8319, 0.8342,  # 0.2 noise
    0.6444, 0.6423, 0.6238, 0.6146,  # 0.4 noise
    0.2996, 0.3160, 0.2930, 0.2786,  # 0.6 noise
]]
mentornet = pd.DataFrame(mentornet, columns=df_results.columns, index = ['mentornet'])
smodel = [[
    0.9375,  # 0 noise
    0.8000, 0.7996, 0.7974, 0.7910,  # 0.2 noise
    0.5856, 0.6121, 0.5913, 0.5752,  # 0.4 noise
    0.2845, 0.2853, 0.2793, 0.2726,  # 0.6 noise
]]
smodel = pd.DataFrame(smodel, columns=df_results.columns, index = ['smodel'])
reed = [[
    0.9372,  # 0 noise
    0.7809, 0.7892, 0.8076, 0.7927,  # 0.2 noise
    0.6048, 0.6041, 0.6124, 0.5860,  # 0.4 noise
    0.2904, 0.2939, 0.2913, 0.2677,  # 0.6 noise
]]
reed = pd.DataFrame(reed, columns=df_results.columns, index = ['reed'])
vanilla = [[
    0.935,  # 0 noise
    0.7843, 0.7916, 0.7901, 0.7825,  # 0.2 noise
    0.6022, 0.6077, 0.5963, 0.5727,  # 0.4 noise
    0.2696, 0.2966, 0.2824, 0.2681,  # 0.6 noise
]]
vanilla = pd.DataFrame(vanilla, columns=df_results.columns, index = ['vanilla'])


# In[47]:


# These are the results if we train our model with batch size 64 for all noise
# rates except 0.4 where we use 32 batch size.
cifar10_final_benchmarks = (df_results / 100).append(mentornet).append(smodel).append(reed).append(vanilla)
cifar10_final_benchmarks.to_csv('cifar10/benchmarks.csv')


# In[89]:


# Final table in the paper
cifar10_final_benchmarks.drop(('0','0'), axis=1)


# In[90]:


# Latex of the final table in the paper
print(cifar10_final_benchmarks.drop(('0','0'), axis=1).round(3).to_latex())


# In[178]:


# These are the results if we train our model with batch size 64 for all noise rates
(df_results / 100).append(mentornet).append(smodel).append(reed).append(vanilla).round(3)


# In[173]:


# These are the results if we train our model with batch size 16, 32 for higher noise rates
(df_results / 100).append(mentornet).append(smodel).append(reed).append(vanilla).round(3)


# In[160]:


# These are the results if we train our model with the pruned, but non-noisy leftover labels
(df_results / 100).append(mentornet).append(smodel).append(reed).append(vanilla).round(3)


# # Benchmarking RMSE estimating the joint

# In[4]:


import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# In[5]:


cifar10_label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[6]:


folders = [c for c in os.listdir(base_dir) if 'noise_amount' in c]
results = []
est_joints = []
true_joints = []
results_argmax = []
experiments = []
rmses = []
for folder in sorted(folders):
    if 'noise_amount__0_8' in folder or 'noise_amount__0_0' in folder:
        continue  # skip
    print(folder)
    psx_file = [z for z in os.listdir(base_dir + folder) if 'pyx' in z][0]
    psx = np.load(base_dir + folder + "/" + psx_file)
    
    #Make sure psx is the right shape
    psx = psx[:,:10]
    
    # Load noisy labels
    frac_zero_noise_rates = folder.split('_')[-7]
    noise_amount = folder.split('_')[-1]
    rfn = 'cifar10_noisy_labels__frac_zero_noise_rates__0.{}__noise_amount__0.{}.json'.format(
        frac_zero_noise_rates, noise_amount)
    with open(noisy_base_dir + "cifar10_noisy_labels/" + rfn, 'r') as rf:
        d = json.load(rf)
    s = np.asarray([v for k,v in d.items()])
    
    # Load the joint used to generate the noisy labels
    rfn = 'cifar10_noise_matrix__frac_zero_noise_rates__0.{}__noise_amount__0.{}.pickle'.format(
        frac_zero_noise_rates, noise_amount)
    with open(noisy_base_dir + "cifar10_noisy_labels/" + rfn, 'rb') as rf:
        nm = pickle.load(rf)
    py = np.bincount(y) / len(y)
    true_joint = nm * py
    
    
    
    true_label_errors = s != y
    acc = np.sum(s == y) / len(y)
    print('accuracy of labels:', acc)
    
    # Estimate the joint with confident learning
    est_joint = cleanlab.latent_estimation.estimate_joint(s, psx)
    
    # Compute noise and cast sparsity to float
    noise = round(1 - acc, 1)
    sparsity = int(frac_zero_noise_rates) / 10.

    # Joint estimation Benchmarks
    cl_rmse = np.sqrt(mean_squared_error(true_joint, est_joint))
    print('RMSE CL: {:.4}'.format(cl_rmse), end = '| ')
    results.append(abs(est_joint - true_joint))
    experiments.append((noise, sparsity))
    est_joints.append(est_joint)
    true_joints.append(true_joint)
    
    # Estimate the joint with confident learning argmax baseline
    cj_argmax = confusion_matrix(np.argmax(psx, axis=1), s).T
    est_joint_argmax = cleanlab.latent_estimation.estimate_joint(s, psx, cj_argmax)
    argmax_rmse = np.sqrt(mean_squared_error(true_joint, est_joint_argmax))
    print('RMSE CL baseline: {:.4}'.format(argmax_rmse), end = '\n\n')
    results_argmax.append(abs(est_joint_argmax - true_joint))
    
    rmses.append({
        'CL RMSE': cl_rmse,
        'CL argmax RMSE': argmax_rmse,
        'Noise': noise,
        'Sparsity': sparsity,
    })


# In[7]:


rmse_for_paper = pd.DataFrame(rmses).set_index(['Noise', 'Sparsity']).sort_index().T.round(3)
print(rmse_for_paper.to_latex())
rmse_for_paper


# In[8]:


def using_multiindex(A, columns):
    shape = A.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({'Absolute Difference': A.flatten()}, index=index).reset_index()
    return df

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(
        index=args[1],
        columns=args[0],
        values=args[2],
    )
    d.index = cifar10_label_names
    d.columns = cifar10_label_names
    ax = sns.heatmap(d,  **kwargs)
#     _ = ax.set(
#         xlabel=r'Latent, true label $y^*$',
#         ylabel=r'Noisy label $\tilde{y}$',
#         title='hey',
#     )
    ax.set_xlabel('', fontsize=20)
    ax.set_ylabel('', fontsize=20)
    ax.set_title('', fontsize=30)
#     ax.set_xticklabels('', fontsize=30)


# In[9]:


grid = using_multiindex(np.stack(results_argmax), ['Z',  r'Latent, true label $y^*$', r'Noisy label $\tilde{y}$', ])
grid = pd.concat([
    pd.DataFrame(
        [z for z in experiments for i in range(100)],
        columns=['Noise', 'Sparsity'],
    ),
    grid,
], axis=1).drop('Z', axis=1)
fg = sns.FacetGrid(grid, row='Noise', col='Sparsity',aspect=1.5, ) #  height=5, 
g = fg.map_dataframe(draw_heatmap, r'Latent, true label $y^*$', r'Noisy label $\tilde{y}$', 'Absolute Difference', cbar=True)
_ = g.set_titles(size=18)

print('\n\nHEATMAP FOR Baseline estimation of joint\n\n')


# In[47]:


grid = using_multiindex(np.stack(results), ['Z',  r'Latent, true label $y^*$', r'Noisy label $\tilde{y}$', ])
grid = pd.concat([
    pd.DataFrame(
        [z for z in experiments for i in range(100)],
        columns=['Noise', 'Sparsity'],
    ),
    grid,
], axis=1).drop('Z', axis=1)
fg = sns.FacetGrid(grid, row='Noise', col='Sparsity', aspect=1.5, ) #  height=5, 
g = fg.map_dataframe(draw_heatmap, r'Latent, true label $y^*$', r'Noisy label $\tilde{y}$', 'Absolute Difference', cbar=True)
_ = g.set_titles(size=18)
plt.savefig('cifar10_abs_diff_ALL.pdf', bbox_inches='tight', pad_inches=0)
print('\n\nCONFIDENT LEARNING estimation of joint\n\n')


# In[11]:


absolute_diff_matrix = results[experiments.index((0.4, 0.6))]
est_joint = est_joints[experiments.index((0.4, 0.6))]
true_joint = true_joints[experiments.index((0.4, 0.6))]


# In[58]:


lol = (true_joint - true_joint.diagonal() * np.eye(len(true_joint))).flatten()
lol.sort()
np.cumsum(lol[::-1])


# In[102]:


sns.set(font_scale=2)
scale_values = 100
savefig = True

plt.figure(figsize = (8,5))
ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
ax = sns.heatmap(
    data=pd.DataFrame(true_joint.round(3) * scale_values, columns=cifar10_label_names, index=cifar10_label_names),
    cbar_kws={'label': 'Joint probability'},
    annot=True,
    vmax=0.1 * scale_values,
    cbar=False,
)
_ = ax.set_xlabel(r'Latent, true label $y^*$', fontsize=30)
_ = ax.set_ylabel(r'Noisy label $\tilde{y}$', fontsize=30)
# _ = ax.set_title(r'True joint (unknown to CL) $ \hat{{Q}}_{\tilde{y}, y^*} $', fontsize=30)
# use matplotlib.colorbar.Colorbar object
# cbar = ax.collections[0].colorbar
# here set the labelsize by 20
# cbar.ax.tick_params(labelsize=13)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 25)
plt.xticks(rotation=45)
if savefig:
    plt.savefig('cifar10_true_joint_noise_4_sparsity_6.pdf', bbox_inches='tight',
    pad_inches=0)

    
plt.figure(figsize = (8,5))
ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
ax = sns.heatmap(
    data=pd.DataFrame(est_joint.round(3) * scale_values, columns=cifar10_label_names, index=cifar10_label_names),
    cbar_kws={'label': 'Joint probability'},
    annot=True,
    vmax=0.1 * scale_values,
    cbar=False,
)
_ = ax.set_xlabel(r'Latent, true label $y^*$', fontsize=30)
# _ = ax.set_ylabel(r'Noisy label $\tilde{y}$', fontsize=20)
# _ = ax.set_title(r'CL estimated joint ${Q}_{\tilde{y}, y^*}$', fontsize=30)
# use matplotlib.colorbar.Colorbar object
# cbar = ax.collections[0].colorbar
# here set the labelsize by 20
# cbar.ax.tick_params(labelsize=13)
ax.figure.axes[-1].yaxis.label.set_size(20)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25)
# ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)
plt.xticks(rotation=45)
plt.yticks([])
if savefig:
    plt.savefig('cifar10_est_joint_noise_4_sparsity_6.pdf', bbox_inches='tight',
    pad_inches=0)

    
plt.figure(figsize = (10,5))
ax = plt.axes([0., 0., 1., 1.], frameon=False, xticks=[],yticks=[])
ax = sns.heatmap(
    data=pd.DataFrame(absolute_diff_matrix.round(3) * scale_values, columns=cifar10_label_names, index=cifar10_label_names),
    cbar_kws={'label': r'Joint probability ($10^{-2}$)'},  # {'label': 'Absolute difference'},
    annot=True,
    vmax=0.1 * scale_values,
)
_ = ax.set_xlabel(r'Latent, true label $y^*$', fontsize=30)
# _ = ax.set_ylabel(r'Noisy label $\tilde{y}$', fontsize=20)
# _ = ax.set_title(r'Absolute difference $ | {Q}_{\tilde{y}, y^*} - \hat{{Q}}_{\tilde{y}, y^*} |$', fontsize=30)
# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=25)
ax.figure.axes[-1].yaxis.label.set_size(30)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 25)
# ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)
plt.xticks(rotation=45)
plt.yticks([])
if savefig:
    plt.savefig('cifar10_abs_diff_noise_4_sparsity_6.pdf', bbox_inches='tight',
    pad_inches=0)


# In[ ]:




