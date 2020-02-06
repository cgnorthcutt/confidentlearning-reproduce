All bash jobs used to generate all confident learning results for the final cifar-10 benchmarking table


PIKEVILLE



cd /home/cgn/coteaching/results/0_2/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_0_2.log ;
cd /home/cgn/coteaching/results/0_2/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_0_2.log ;
cd /home/cgn/coteaching/results/0_2/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_0_2.log ;
cd /home/cgn/coteaching/results/0_2/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_0_2.log ;
cd /home/cgn/coteaching/results/0_2/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_0_2.log ;

cd /home/cgn/coteaching/results/2_2/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_2_2.log ;
cd /home/cgn/coteaching/results/2_2/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_2_2.log ;
cd /home/cgn/coteaching/results/2_2/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_2_2.log ;
cd /home/cgn/coteaching/results/2_2/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_2_2.log ;
cd /home/cgn/coteaching/results/2_2/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_2_2.log ;

cd /home/cgn/coteaching/results/4_2/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_4_2.log ;
cd /home/cgn/coteaching/results/4_2/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_4_2.log ;
cd /home/cgn/coteaching/results/4_2/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_4_2.log ;
cd /home/cgn/coteaching/results/4_2/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_4_2.log ;
cd /home/cgn/coteaching/results/4_2/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_4_2.log ;

cd /home/cgn/coteaching/results/6_2/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_6_2.log ;
cd /home/cgn/coteaching/results/6_2/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_6_2.log ;
cd /home/cgn/coteaching/results/6_2/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_6_2.log ;
cd /home/cgn/coteaching/results/6_2/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_6_2.log ;
cd /home/cgn/coteaching/results/6_2/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json --train_mask_dir train_mask.npy ; } &> out_6_2.log ;

MOREHEAD



cd /home/cgn/coteaching/results/0_4/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_0_4.log ;
cd /home/cgn/coteaching/results/0_4/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_0_4.log ;
cd /home/cgn/coteaching/results/0_4/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_0_4.log ;
cd /home/cgn/coteaching/results/0_4/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_0_4.log ;
cd /home/cgn/coteaching/results/0_4/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_0_4.log ;

cd /home/cgn/coteaching/results/2_4/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_2_4.log ;
cd /home/cgn/coteaching/results/2_4/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_2_4.log ;
cd /home/cgn/coteaching/results/2_4/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_2_4.log ;
cd /home/cgn/coteaching/results/2_4/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_2_4.log ;
cd /home/cgn/coteaching/results/2_4/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_2_4.log ;

cd /home/cgn/coteaching/results/4_4/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_4_4.log ;
cd /home/cgn/coteaching/results/4_4/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_4_4.log ;
cd /home/cgn/coteaching/results/4_4/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_4_4.log ;
cd /home/cgn/coteaching/results/4_4/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_4_4.log ;
cd /home/cgn/coteaching/results/4_4/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_4_4.log ;

cd /home/cgn/coteaching/results/6_4/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_6_4.log ;
cd /home/cgn/coteaching/results/6_4/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_6_4.log ;
cd /home/cgn/coteaching/results/6_4/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_6_4.log ;
cd /home/cgn/coteaching/results/6_4/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_6_4.log ;
cd /home/cgn/coteaching/results/6_4/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json --train_mask_dir train_mask.npy ; } &> out_6_4.log ;

RICHMOND


cd /home/cgn/coteaching/results/0_6/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_0_6.log ;
cd /home/cgn/coteaching/results/0_6/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_0_6.log ;
cd /home/cgn/coteaching/results/0_6/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_0_6.log ;
cd /home/cgn/coteaching/results/0_6/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_0_6.log ;
cd /home/cgn/coteaching/results/0_6/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=0 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_0_6.log ;

cd /home/cgn/coteaching/results/2_6/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_2_6.log ;
cd /home/cgn/coteaching/results/2_6/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_2_6.log ;
cd /home/cgn/coteaching/results/2_6/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_2_6.log ;
cd /home/cgn/coteaching/results/2_6/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_2_6.log ;
cd /home/cgn/coteaching/results/2_6/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=1 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_2_6.log ;

cd /home/cgn/coteaching/results/4_6/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_4_6.log ;
cd /home/cgn/coteaching/results/4_6/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_4_6.log ;
cd /home/cgn/coteaching/results/4_6/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_4_6.log ;
cd /home/cgn/coteaching/results/4_6/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_4_6.log ;
cd /home/cgn/coteaching/results/4_6/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=2 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_4_6.log ;

cd /home/cgn/coteaching/results/6_6/train_pruned_cl_pbnr ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_6_6.log ;
cd /home/cgn/coteaching/results/6_6/train_pruned_cl_pbc ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_6_6.log ;
cd /home/cgn/coteaching/results/6_6/train_pruned_cl_both ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_6_6.log ;
cd /home/cgn/coteaching/results/6_6/train_pruned_argmax ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_6_6.log ;
cd /home/cgn/coteaching/results/6_6/train_pruned_conf_joint_only ;
{ time CUDA_VISIBLE_DEVICES=3 python2 /home/cgn/cgn/confidentlearning-reproduce/other_methods/coteaching/main.py --fn /datasets/datasets/cifar10/cifar10/noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json --train_mask_dir train_mask.npy ; } &> out_6_6.log ;

