cd ~/sceloss_results
mkdir 0_2
cd 0_2
CUDA_VISIBLE_DEVICES=0 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.2.json' > out_0_2.log

cd ~/sceloss_results
mkdir 2_2
cd 2_2
CUDA_VISIBLE_DEVICES=1 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.2.json' > out_2_2.log

cd ~/sceloss_results
mkdir 4_2
cd 4_2
CUDA_VISIBLE_DEVICES=2 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.2.json' > out_4_2.log


mkdir ~/sceloss_results
cd ~/sceloss_results
mkdir 6_2
cd 6_2
CUDA_VISIBLE_DEVICES=0 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.2.json' > out_6_2.log

cd ~/sceloss_results
mkdir 0_4
cd 0_4
CUDA_VISIBLE_DEVICES=1 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.4.json' > out_0_4.log

cd ~/sceloss_results
mkdir 2_4
cd 2_4
CUDA_VISIBLE_DEVICES=2 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.4.json' > out_2_4.log

cd ~/sceloss_results
mkdir 4_4
cd 4_4
CUDA_VISIBLE_DEVICES=3 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json' > out_4_4.log


cd ~/sceloss_results
mkdir 6_4
cd 6_4
CUDA_VISIBLE_DEVICES=0 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json' > out_6_4.log

cd ~/sceloss_results
mkdir 0_6
cd 0_6
CUDA_VISIBLE_DEVICES=1 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.0__noise_amount__0.6.json' > out_0_6.log

cd ~/sceloss_results
mkdir 2_6
cd 2_6
CUDA_VISIBLE_DEVICES=2 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.2__noise_amount__0.6.json' > out_2_6.log

cd ~/sceloss_results
mkdir 4_6
cd 4_6
CUDA_VISIBLE_DEVICES=3 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.6.json' > out_4_6.log



cd ~/sceloss_results
mkdir 6_6
cd 6_6
CUDA_VISIBLE_DEVICES=0 python3 ~/Dropbox\ \(MIT\)/cgn/SCELoss-Reproduce/train.py --fn '/home/cgn/Dropbox (MIT)/cgn/cleanlab/examples/cifar10/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.6.json' > out_6_6.log

