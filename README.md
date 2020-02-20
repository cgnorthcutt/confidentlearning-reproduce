# confidentlearning-reproduce
Experimental data for reproducibility of CIFAR-10 experimental results in the [confident learning paper](https://arxiv.org/abs/1911.00068).

The code to generate these Confident Learning CIFAR-10 benchmarking results is available in the [**`cleanlab`**](https://pypi.org/project/cleanlab/) Python package, specifically in [`cleanlab/examples/cifar10/`](https://github.com/cgnorthcutt/cleanlab/tree/master/examples/cifar10).

Because GitHub limits filesizes to 100MB, I cannot upload trained ResNet-50 models (180MB each), but for every setting, I upload an `out` log file with the accuracy at every batch and test accuracy at every epoch. The file naming conventions are as follows

* `out` -- the log files during training
* `train_mask.npy` -- boolean vector for which examples where pruned during training
* `cifar10__train__model_resnet50__pyx.npy` -- Cross-validation out of sample predicted probabilities for CIFAR-10 under the given noisy labels settings
* `cifar10_noisy_labels` -- folder containing all the noisy labels settings
* `experiments.bash` -- examples of the commands run to generate results
* `cifar10_train_crossval.py` -- training script to perform all cifar-10 experiments (get cross-validated probabilities, evaluate on test set, train on a masked input to remove noisy examples)

## Need out-of-sample predicted probabilities for CIFAR-10 train set?

You can obtain standard (no noise added to label) predicted probabilities [here](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_0/cifar10__train__model_resnet50__pyx.npy).

These are computed using four-fold cross-validation with a ResNet50 architecture. You can download the out-of-sample predicted probabilities for all training examples in CIFAR-10 for various noise and sparsities settings here:
 * Noise: 0% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_0/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]





## License

Copyright (c) 2017-2020 Curtis Northcutt. Released under the MIT License. See [LICENSE](https://github.com/cgnorthcutt/cleanlab/blob/master/LICENSE) for details.
