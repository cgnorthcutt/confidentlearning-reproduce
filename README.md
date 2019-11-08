# confidentlearning-data
Experimental data for reproducibility of CIFAR-10 experimental results in the [confident learning paper](https://arxiv.org/abs/1911.00068).

Because GitHub limits filesizes to 100MB, I cannot upload trained ResNet-50 models (180MB each), but for every setting, I upload an `out` log file with the accuracy at every batch and test accuracy at every epoch. The file naming conventions are as follows

* `out` -- the log files during training
* `train_mask.npy` -- boolean vector for which examples where pruned during training
* `cifar10__train__model_resnet50__pyx.npy` -- Cross-validation out of sample predicted probabilities for CIFAR-10 under the given noisy labels settings
* `cifar10_noisy_labels` -- folder containing all the noisy labels settings
* `experiments.bash` -- examples of the commands run to generate results
* `cifar10_train_crossval.py` -- training script to perform all cifar-10 experiments (get cross-validated probabilities, evaluate on test set, train on a masked input to remove noisy examples)
