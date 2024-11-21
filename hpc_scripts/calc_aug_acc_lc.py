import time
import cv2
import random
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Subset

import syn_paper_utils as utils

# Option for whether to only plot the results from saved scores, or to recalculate from scratch.
# Note this is very slow, but allows reproduction of all results
recalculate_results = True

device_name = utils.get_device()
device = torch.device(device_name)
CNN_params_dict = utils.CNN_params_setup(device)
dropout_conv, dropout_fc = CNN_params_dict['dropout_conv'], CNN_params_dict['dropout_fc']

if recalculate_results: # Load the CIFAR-10 data to memory
    full_cifar_dataset = CIFAR10
    # Define data transformations
    transform = transforms.Compose([transforms.ToTensor(),]) # Convert PIL Image to PyTorch Tensor

    train_dataset = full_cifar_dataset(
        root=f'./{"data/cifar-10-orig"}',  # Change the root directory as needed
        train=True,      # Set to True for the training set
        transform=transform,
        download=True)

    test_dataset = full_cifar_dataset(
        root=f'./{"data/cifar-10-orig"}',  # Change the root directory as needed
        train=False,     # Set to False for the test set
        transform=transform,
        download=True)
    
# Setup our main n_orig = 1024 sample
N = 1024

seed = 42
random.seed(seed)

random_indices = random.sample(range(len(train_dataset)), N)
train_samp = Subset(train_dataset, random_indices)
# Separate the images and labels
images = torch.stack([train_samp[i][0] for i in range(N)])  # Stack the image tensors
labels = torch.tensor([train_samp[i][1] for i in range(N)])  # Convert labels to a tensor

# Return as TensorDataset
subset_train_dataset = TensorDataset(images, labels)

gamma = 2.5
alphas = [None, 2, 5, 20, 50]
n_samp = [100, 200, 500, 1024, 2000, 4000, 8000, 15000, 30000, 50000]

aug_lc_dict_json_name = "scores/aug_acc_lc.json"

if recalculate_results:
    train_length = len(train_dataset)
    seed = 42
    random.seed(seed)
    aug_lc_dict = {}
    # Loop through the alphas (amount of augmented data to use)
    for j, alpha in enumerate(alphas):
        # Set up name and aug_lc_dict for this iteration depending on augmentation type
        aug_bool = alpha is not None
        if aug_bool:
            score_name = f"alpha = {alpha}"
        else:
            score_name = "No augmentation"
        aug_lc_dict[score_name] = {}

        # Loop through the data sample lengths
        for i, n in enumerate(n_samp):
            # Time the code
            start_time = time.time()
            
            # Sample the cifar data n times (n being the current number of samples for this iteration) without replacement
            random_indices = random.sample(range(train_length), n)
            # Organise into a dataset
            random_reals = [train_dataset[i] for i in random_indices]
            random_x, random_y = zip(*random_reals)  # Unzip data into separate lists
            lc_subset_train_dataset = TensorDataset(torch.stack(random_x), torch.tensor(random_y))
            
            cnn_mod = utils.Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)
            _, _, test_acc = utils.nn_trainer(cnn_mod, lc_subset_train_dataset, test_dataset, opt_type = "adam", CNN_params_dict=CNN_params_dict, 
                                                        loss_type = "nll", lr_sched = None, device_str = device_name, verbose = False, 
                                                        augmentation = aug_bool, aug_ratio = alpha, aug_var = gamma)
            aug_lc_dict[score_name][n] = test_acc  
            end_time = time.time()
            print(f"Run {(i+1) + (j)*(len(n_samp))} out of {len(n_samp) * len(alphas)} complete, for {n} samples and {score_name}. Test score = {round(test_acc, 3)}. Runtime = {round(end_time - start_time, 1)} seconds")

            # Save the output dictionary to json file
            with open(aug_lc_dict_json_name, 'w') as json_file:
                json.dump(aug_lc_dict, json_file)
else:
    with open(aug_lc_dict_json_name, 'r') as json_file:
        aug_lc_dict = json.load(json_file)