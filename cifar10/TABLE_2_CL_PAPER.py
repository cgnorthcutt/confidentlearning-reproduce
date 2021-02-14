#!/usr/bin/env python
# coding: utf-8

# # Table 2 CIFAR-10 Benchmarking (10 trials for each table entry) in Confident Learning paper.

# # Benchmarking Confident Learning with TEN TRIALS EACH
# * uses co-teaching
# * learning with noisy labels accuracy on CIFAR-10

# In[1]:


import pandas as pd


# In[2]:


base = '/home/cgn/cifar10_cl_coteaching/'
experiments = ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']
num_trials = 10


# In[3]:


dfs = {}
for seed in range(1, num_trials + 1):
    results = []
    for experiment in ['argmax', 'pbc', 'pbnr', 'both', 'cj_only']:
        for noise in [2, 4, 6]:
            for sparsity in [0, 2, 4, 6]:
                f = str(sparsity) + "_" + str(noise)
                path = base + "trial{}/".format(seed) + experiment + "/" + f + "/out.log"
                try:
                    with open(path, 'r') as rf:
                        lines = rf.readlines()
                    acc = [[float(z[3:-1]) for z in l.split('Acc')[1:]] for l in lines if l.startswith(' * Acc')]
                    model1, model2 = acc[::2], acc[1::2]
                    acc1 = max([max(model2[i][0], a[0]) for i, a in enumerate(model1)])
                    acc5 = max([max(model2[i][1], a[1]) for i, a in enumerate(model1)])
                except:
                    print(path)
                    print(" ".join(lines[-5:-4]))
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
    dfs[seed] = df_results
    
cl = pd.concat(dfs.values()).mean(level=0)
cl_std = pd.concat(dfs.values()).std(level=0)
cl = cl.reindex(['argmax', 'pbc', 'cj_only', 'both', 'pbnr'])
cl_std = cl_std.reindex(['argmax', 'pbc', 'cj_only', 'both', 'pbnr'])
cl.round(2)


# ## Results on other models -- ran on Google's servers by co-author Lu Jiang

# In[4]:


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


# In[5]:


# With standard deviations
cl.round(1).astype(str) + '±' + cl_std.round(1).astype(str)


# In[6]:


# Final table of results.
cifar10_final_benchmarks = (cl / 100).append(mentornet).append(smodel).append(reed).append(vanilla) * 100
cifar10_final_benchmarks.round(1)


# In[7]:


# Latex of the final table in the paper
method_name_map = {
	'argmax': r'CL: $\bm{C}_{\text{confusion}}$',
	'pbc': 'CL: PBC',
	'cj\_only': r'CL: $\cj$',
	'both': 'CL: C+NR',
	'pbnr': 'CL: PBNR',
	'vanilla': 'Baseline',
}
tex = cifar10_final_benchmarks.round(1).to_latex().replace('±', ' \pm ').replace('nan', '0.1')
for k, v in method_name_map.items():
    tex = tex.replace(k, v)
print(tex)


# In[28]:


print('Mean std. dev. of CL methods across all sparsities for:')
print(' * 20% noise: {:.2f}%'.format(cl_std.iloc[:,:4].mean().mean()))
print(' * 40% noise: {:.2f}%'.format(cl_std.iloc[:,4:8].mean().mean()))
print(' * 70% noise: {:.2f}%'.format(cl_std.iloc[:,8:12].mean().mean()))


# In[10]:


tex = cl_std.round(2).to_latex()
for k, v in method_name_map.items():
    tex = tex.replace(k, v)
print(tex)

