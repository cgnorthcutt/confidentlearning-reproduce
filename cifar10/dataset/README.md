# CIFAR-10 dataset prepared for PyTorch

This folder contains the CIFAR-10 dataset prepared for PyTorch. There are two folders:
- test: Contains ten folders, one for each class, each containing approx 1000 images
- train: Contains ten folders, one for each class, each containng approx 5000 images.

The train and test folders have been zipped for ease of downloading. The train set has been broken into two files to allow uploading to GitHub.
To unzip the folders, run:

tar -xzvf train_part_1_of_2.tar.gz;
tar -xzvf train_part_2_of_2.tar.gz;
tar -zxvf test.tar.gz

or the easiest way is to just type, in terminal:
./prepare_dataset.bash

Which will run those above commands for you.

