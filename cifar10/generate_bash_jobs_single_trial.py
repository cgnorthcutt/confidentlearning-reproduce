#!/usr/bin/env python
# coding: utf-8

# # Script to generate CIFAR-10 benchmarking jobs

# In[24]:


template = '''mkdir -p /home/cgn/cifar10_cl_coteaching/final/{method}/{sparsity}_{noise}
cd /home/cgn/cifar10_cl_coteaching/final/{method}/{sparsity}_{noise}
{{ time python3 ~/cgn/cleanlab/examples/cifar10/cifar10_train_crossval.py --coteaching --forget-rate 0.{forget_rate} --seed 1 --batch-size {batchsize} --lr {lr} --epochs 350 \
--train-labels "${{base}}/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.{sparsity}__noise_amount__0.{noise}.json" --gpu {gpu} \
--dir-train-mask "${{base}}/cifar10_noisy_labels__frac_zero_noise_rates__0_{sparsity}__noise_amount__0_{noise}/train_pruned_{method_long}/train_mask.npy" \
/datasets/datasets/cifar10/cifar10/ ; }} &> out.log
'''

keys = ['argmax', 'cj_only', 'pbc', 'pbnr', 'both']
vals = ['argmax', 'conf_joint_only', 'cl_pbc', 'cl_pbnr', 'cl_both']
methods = dict(zip(keys, vals))
total_gpus = 12
num_jobs = 60
jobs_per_gpu = int(round(num_jobs / total_gpus))
gpus_per_machine = 4
batchsize = 256
lr = 0.001

counter = 0
for sparsity in [0, 2, 4, 6]:
    for noise in [2, 4, 6]:
        for method, method_long in methods.items():
            if counter % jobs_per_gpu == 0:
                print('\nbase="/home/cgn/cgn/confidentlearning-reproduce/cifar10"')
            
            print(template.format(
                batchsize=batchsize,
                lr=lr,
                sparsity=sparsity,
                noise=noise,
                forget_rate=noise//2,
                method=method,
                method_long=method_long,
                gpu=counter // jobs_per_gpu % gpus_per_machine,
            ))
            counter += 1
            


# In[ ]:




