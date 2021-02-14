#!/usr/bin/env python
# coding: utf-8

# # Script to generate CIFAR-10 benchmarking jobs

# In[13]:


import pandas as pd
df_batch_size = pd.read_pickle('batch_sizes_df.p')


# In[14]:


template = '''mkdir -p /home/cgn/cifar10_cl_coteaching/trial{seed}/{method}/{sparsity}_{noise}
cd /home/cgn/cifar10_cl_coteaching/trial{seed}/{method}/{sparsity}_{noise}
{{ time python3 ~/cgn/cleanlab/examples/cifar10/cifar10_train_crossval.py \
--coteaching --forget-rate 0.{forget_rate} --seed {seed} --batch-size {batchsize} \
--lr {lr} --epochs {epochs} --gpu {gpu} \
--train-labels "${{base}}/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.{sparsity}__noise_amount__0.{noise}.json" \
--dir-train-mask "${{base}}/cifar10_noisy_labels__frac_zero_noise_rates__0_{sparsity}__noise_amount__0_{noise}/train_pruned_{method_long}/train_mask.npy" \
/datasets/datasets/cifar10/cifar10/ ; }} &> out.log
'''

gpu_rig_names = ['Pikeville', 'Richmond', 'Morehead', 'Kentucky']
keys = ['argmax', 'cj_only', 'pbc', 'pbnr', 'both']
vals = ['argmax', 'conf_joint_only', 'cl_pbc', 'cl_pbnr', 'cl_both']
sparsities = [0, 2, 4, 6]
noise_rates = [2, 4, 6]
methods = dict(zip(keys, vals))
total_gpus = 16
num_trials = 10
start_trial = 2
trials = range(start_trial, num_trials + 1)  # Trials are 1-indexed.
num_jobs = len(methods) * len(sparsities) * len(noise_rates) * len(trials)
jobs_per_gpu = int(round(num_jobs / total_gpus))
gpus_per_machine = 4
lr = 0.001
epochs = 350

counter = 0
gpu_counter = 0
for sparsity in sparsities:
    for noise in noise_rates:
        for method, method_long in methods.items():
            # Index by # (noise_amount, frac_zero_noise_rates)
            batch_size = df_batch_size[(noise / 10., sparsity / 10.)][method]
            for seed in trials:  # 1-indexed trials
                gpu_id = counter // jobs_per_gpu % gpus_per_machine
                if counter % jobs_per_gpu == 0:
                    rig_name = gpu_rig_names[gpu_counter // gpus_per_machine]
                    print('\n# Machine: {} | GPU: {}\n'.format(rig_name, gpu_id))
                    print('base="/home/cgn/cgn/confidentlearning-reproduce/cifar10"')
                    gpu_counter += 1

                print('# Job: {}'.format(counter % jobs_per_gpu + 1))
                print(template.format(
                    batchsize=batch_size,
                    lr=lr,
                    sparsity=sparsity,
                    noise=noise,
                    forget_rate=noise//2,
                    method=method,
                    method_long=method_long,
                    gpu=gpu_id,
                    epochs=epochs,
                    seed=seed,
                ))
                counter += 1
assert(counter == num_jobs)


# In[ ]:




