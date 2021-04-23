# Copyright (c) 2017-2050 Curtis G. Northcutt
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.

# coding: utf-8
# Author: Curtis G. Northcutt, MIT

'''A simple script to ensure pyx np matrices are correct size: (50,000, 10).'''


import numpy as np
import os


# Fix shape of pyx matrices if needed.
for f in os.listdir():
    if 'cifar10_noisy_labels__frac_zero_noise_rates__' in f:
        with open(f + '/cifar10__train__model_resnet50__pyx.npy', 'rb') as rf:
            pyx = np.load(rf)
        with open(f + '/cifar10__train__model_resnet50__pyx.npy', 'wb') as wf:
            np.save(wf, pyx[:, :10])


# Make sure the shape is correct.
for f in os.listdir():
    if 'cifar10_noisy_labels__frac_zero_noise_rates__' in f:
        with open(f + '/cifar10__train__model_resnet50__pyx.npy', 'rb') as rf:
            pyx = np.load(rf)
        assert pyx.shape == (50000, 10)



