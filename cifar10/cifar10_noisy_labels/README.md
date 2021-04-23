# Noisy labels used in confident learning experiments

This folder contains the noisy labels using in the cifar-10 expeirments for confident learning as well as the noise matrices used to generate the noisy labels.

In these files `frac_zero_noise_rates` is the same thing as `sparsity`, as it is referred to in the [confident learning paper](https://www.jair.org/index.php/jair/article/view/12125).

## How to load the noisy labels

The noisy labels are stored as json files. As an example, to load a json file:

```python
import json
with open('cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json', 'r') as rf:
  noisy_labels = json.load(rf)  # this loads a dict, mapping filename (str) --> label (int)
```

The first few rows of `noisy_labels` from the code above, should look like this:

```python
>>> for i, (k, v) in enumerate(noisy_labels.items()):
>>>      if i < 5:
>>>          print("'" + k + "': {},".format(v))
...
'/datasets/datasets/cifar10/cifar10/train/airplane/10008_airplane.png': 0,
'/datasets/datasets/cifar10/cifar10/train/airplane/10010_airplane.png': 1,
'/datasets/datasets/cifar10/cifar10/train/airplane/10020_airplane.png': 8,
'/datasets/datasets/cifar10/cifar10/train/airplane/10024_airplane.png': 5,
'/datasets/datasets/cifar10/cifar10/train/airplane/10031_airplane.png': 5,
```

## How to load the noise matrices

The noisy labels are stored as json files. As an example, to load a json file:

```python
import numpy as np
noise_matrix = np.load(  # this loads a np.array 2d matrix of floats
    'cifar10_noise_matrix__frac_zero_noise_rates__0.0__noise_amount__0.2.pickle',
    allow_pickle=True,
) 
```

The first few rows of `noise_matrix` from the code above, should look like this:

```python
>>> print(noise_matrix[:2, :2])
[[0.53284782 0.00553727]
 [0.07205823 0.84086117]]
```

Note, the noise matrices are not needed to reproduce any results.
You can just use the noisy labels from the json files.
We provide the noise matrices purely for transparency into how the noisy labels were generated.
