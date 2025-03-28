import os
import re
import warnings
import string
import cv2
import numpy as np
import random
import tqdm
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import random_split, DataLoader, TensorDataset, RandomSampler, ConcatDataset, Subset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling

from scipy import linalg
from sklearn.linear_model import LinearRegression
from torch.nn import Parameter as P
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3
from torchvision.models.inception import inception_v3

from img_transformation import img_transformation

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
def data_sampler(train_data, n_samp, specific_ind = [], rand_seed = 42):
    """Take a random sample from the training dataset. Allows specific item selection if desired
    Args:
        train_data (torch dataset): the training data
        n_samp (int): number of samples desired
        specific_ind (list of int, optional): any specific indexes to by selected. Defaults to [].
        rand_seed (int, optional): initial seeding of random sampling. Defaults to 42.
    Raises:
        ValueError: if n_samp is too high (specifically if n_samp + len(specific_ind) > len(train_data))
        ValueError: if specific_ind is not a list of integers
    Returns:
        train_samp: The sampled training data
    """
    n_data = len(train_data)
    len_spec = len(specific_ind)
    if n_data < n_samp + len_spec:
        raise ValueError(f"n_samp ({n_samp}) too high for the given dataset.")
    sample_list = []
    if len_spec != 0: # specific_ind allows the user to select specific images they would like to be added to the sample (e.g. to replicate specific results)
        if isinstance(specific_ind, list) and all(isinstance(item, int) for item in specific_ind): #checks that this is a list of integers
            sample_list += specific_ind
        else:
            raise ValueError(f"specific_ind ({specific_ind}) needs to be a list of integers (indexes of specific images to be extracted), or an empty list.")
    random.seed(rand_seed) # keeps sampling reproducible
    random_indices = random.sample(range(n_data), n_samp - len_spec)
    sample_list += random_indices
    train_samp = [train_data[i] for i in sample_list]
    return train_samp


def train_sampler(train_dataset, n_samp, seed = 42):
    """ Short function to sample training data and return the sampled dataset and a separate tensor for the x data and a list of the y data """
    train_samp = data_sampler(train_dataset, n_samp, rand_seed = seed) # function from igpt_generator file
    x_train_true = [item[0] for item in train_samp]
    y_train_true = [item[1] for item in train_samp]
    images_true = torch.stack(x_train_true)
    return train_samp, images_true, y_train_true 

    
def CNN_params_setup(device):
    """ Function to get a dict of hyperparameters for a selected CNN model.
    Args:
        device (torch device): The torch device in use
    Returns:
        dict: Dictionary of CNN hyperparameters
    """
    CNN_params = {}
    # preset some key hyperparameters for the CNN model
    epochs = 80
    batch_size = 64
    learning_rate = 0.0003
    val_split = 0 #0.2
    weight_decay = 5e-5
    dropout_conv = 0.3
    dropout_fc = 0.5
    # add version-specific parameters to the output
    CNN_params["dropout_conv"] = dropout_conv
    CNN_params["dropout_fc"] = dropout_fc
    CNN_params["epochs"] = epochs
    CNN_params["batch_size"] = batch_size
    CNN_params["learning_rate"] = learning_rate
    CNN_params["val_split"] = val_split 
    CNN_params["weight_decay"] = weight_decay
    CNN_params["device"] = device
    return CNN_params

def conv_augmenter(train_data, aug_ratio = 2, aug_var = 3, return_combined = True, labelled_data = True, random_erase_greyscale = True):
    """Function to carry out conventional image augmentation. Uses `img_transformation` function.
    Args:
        train_data (Dataset): Dataset of (real) training images.
        aug_ratio (float, optional): The proportion of augmented images (to real images) to create. Must be >= 1. Defaults to 2.
        aug_var (float, optional): The transform variability parameter gamma to use, typically a value between 0 and 5 (0 for no augmentations). Defaults to 3.
        return_combined (Bool, optional): Whether to return combined real & augmented data, or the augmented data only. Defaults to True.
        labelled_data (Bool, optional): Whether handling data with labels, or images only. Defaults to True.
        random_erase_greyscale (Bool, optional): Whether to include random erasures & greyscaling into the image transformations. Defaults to True.
    Returns:
        aug_train_data (Dataset): Combined real and augmented datasets
    """
    train_size = len(train_data)
    aug_size = int(np.floor(train_size * aug_ratio - train_size))
    aug_imgs = []
    aug_labels = []

    random_sampler = RandomSampler(train_data, replacement=False, num_samples=aug_size)
    aug_dataloader = DataLoader(train_data, batch_size=1, sampler=random_sampler)

    if labelled_data:
        for img, label in aug_dataloader:
            new_img = img_transformation(img, prob = 1, gamma = aug_var, random_erase_greyscale = random_erase_greyscale)
            aug_imgs += [new_img.squeeze(0)]; aug_labels += [label]
        aug_data = TensorDataset(torch.stack(aug_imgs), torch.tensor(aug_labels))
    else:
        for img in aug_dataloader:
            new_img = img_transformation(img[0], prob = 1, gamma = aug_var, random_erase_greyscale = random_erase_greyscale)
            aug_imgs += [new_img.squeeze(0)]
        aug_data = TensorDataset(torch.stack(aug_imgs))

    if return_combined:
        output_dataset = ConcatDataset([train_data, aug_data])
    else:
        output_dataset = aug_data
    return output_dataset
    
class Cifar_CNN(Module):
    # Based on the following architecture:
    #https://www.kaggle.com/code/sid2412/cifar10-cnn-model-85-97-accuracy#
    def __init__(self, num_channels, classes, k_conv = 3, k_pool = 2, dropout_conv = 0.3, dropout_fc = 0.5):
        from torch.nn import Conv2d, ReLU, Dropout, MaxPool2d, Linear, LogSoftmax
        # Call the parent constructor
        super(Cifar_CNN, self).__init__()
        # initialize first set of CONV -> RELU -> POOL layers
        self.conv1 = Conv2d(in_channels = num_channels, out_channels = 128, kernel_size = k_conv, padding = 1)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size = k_pool, stride = k_pool)
        self.dropout1 = Dropout(dropout_conv)
        # initialize second set of CONV -> RELU -> POOL layers
        self.conv2 = Conv2d(in_channels = 128, out_channels = 256, kernel_size = k_conv, padding = 1)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size = k_pool, stride = k_pool)
        self.dropout2 = Dropout(dropout_conv)
        # initialize set of 3 CONV -> RELU layers, with dropout and pooling at the end
        self.conv3 = Conv2d(in_channels = 256, out_channels = 512, kernel_size = k_conv, padding = 1)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = k_conv, padding = 1)
        self.relu4 = ReLU()
        self.conv5 = Conv2d(in_channels = 512, out_channels = 256, kernel_size = k_conv, padding = 1)
        self.relu5 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size = k_pool, stride = k_pool)
        self.dropout3 = Dropout(dropout_conv)
        # initialise 3 sets of FC -> RELU layers (FC = fully connected == dense layer)
        self.fc1 = Linear(in_features = 256 * 4 * 4, out_features = 512)
        self.relu6 = ReLU()
        self.dropout4 = Dropout(dropout_fc)
        self.fc2 = Linear(in_features = 512, out_features = 256)
        self.relu7 = ReLU()
        self.dropout5 = Dropout(dropout_fc)
        self.fc3 = Linear(in_features = 256, out_features = 128)
        self.relu8 = ReLU()
        self.dropout6 = Dropout(dropout_fc)
        # initialize our softmax classifier
        self.fc4 = Linear(in_features = 128, out_features = classes)
        self.log_soft_max = LogSoftmax(dim = 1)

    def forward(self, x):
        from torch import flatten
        # pass the input through our first set of CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        # pass the output from the previous layer through the second set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        # now for the triple set of conv relu layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        # flatten the output from the previous layer and pass it through our sets of FC => RELU layers
        x = flatten(x, 1) # From multi-dimensional tensor to 1D list of values
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout5(x)
        x = self.fc3(x)
        x = self.relu8(x)
        x = self.dropout6(x)
        # pass the output to our softmax classifier to get our output predictions
        x = self.fc4(x)
        output = self.log_soft_max(x)

        # return the output predictions
        return output
    
def nn_model_train_step(model, train_data_loader, device, loss_fn, opt):
    """ Utility function to carry out the training step of the nn_trainer 
    Args:
        model (PyTorch model): Input neural network model to be trained
        train_data_loader (PyTorch dataloader): Dataloader for the training data
        device (str): Device to be used for model training.
        loss_fn (PyTorch function): loss function
        opt (PyTorch function): optimiser function
    Returns:
        model (PyTorch model): Final trained neural network model.
        total_train_loss (float): total training loss
        train_accuracy (float): training accuracy
    """
    # set model in training mode
    model.train()
    # initialize the total training loss and the number of correct predictions in the training step
    total_train_loss = 0; train_correct_count = 0
    # loop over the training set
    for (x, y) in train_data_loader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # zero out the gradients, perform the backpropagation step, and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to the total training loss so far and calculate the number of correct predictions
        total_train_loss += loss.item()
        train_correct_count += (pred.max(1).indices == y).sum().item()
    return model, total_train_loss, train_correct_count / len(train_data_loader.dataset)

def nn_model_eval_step(model, val_data_loader, device, loss_fn):
    """ Utility function to carry out the evaluation step of the nn_trainer
    Args:
        model (PyTorch model): Input neural network model to be trained
        val_data_loader (PyTorch dataloader): Dataloader for the validation data        
        device (str): Device to be used for model training.
        loss_fn (PyTorch function): loss function
    Returns:
        model (PyTorch model): Final trained neural network model.
        total_val_loss (float): total validation loss
        val_accuracy (float): validation accuracy
        """
    # set model in evaluation mode
    model.eval()
    # initialize the total validation loss and the number of correct predictions in the validation step
    total_val_loss = 0; val_correct_count = 0
    # switch off autograd for evaluation
    with torch.no_grad():
        # loop over the validation set
        for (x, y) in val_data_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            
            # make the predictions and calculate the validation loss
            pred = model(x)
            total_val_loss += loss_fn(pred, y).item()
            # calculate the number of correct predictions
            val_correct_count += (pred.max(1).indices == y).sum().item()
    return model, total_val_loss, val_correct_count / len(val_data_loader.dataset)

def nn_trainer(model, train_data, test_data = None, opt_type = "adam", loss_type = "nll", val_split = 0.2, 
               val_each_step = False, epochs = 50, batch_size = 64, learning_rate = 1e-3, lr_sched = None, 
               augmentation = False, aug_ratio = 2, aug_var = 3, weight_decay = 0, 
               device_str = "cpu", verbose = True, CNN_params_dict = None):
    """ Function to train a neural network model
    Args:
        model (PyTorch model): Input neural network model to be trained. Note this has already been typically sent 'to' the device to be used for model training.
        train_data (dataset): Dataset of training images
        test_data (dataset): Dataset of testing images. Set to None to skip calculating a separate test score. Defaults to None.
        opt_type (str, optional): Name of optimiser function to use. Currently only takes "adam". Defaults to "adam".
        loss_type (str, optional): Name of loss function to use. Currently only takes "nll". Defaults to "nll".
        val_split (float, optional): Proportion of data to use for the validation set (and 1 - val_split for the test set) Overwritten if k_fold_k used. Set to 0 for no validation. Defaults to 0.2.
        val_each_step (bool, optional): Whether to calculate statistics on the validation at each step/epoch; if False, only calculate at the end. Defaults to False.
        epochs (int, optional): Number of epochs to train over. Defaults to 50.
        batch_size (int, optional): Size of batches to split the data into for training. Defaults to 64.
        learning_rate (float, optional): Learning rate parameter for the optimiser. Defaults to 1e-3.
        lr_sched (str, optional): If not None, type of learning rate lr_scheduler (currently only takes None or not for a ReduceLROnPlateau). Defaults to None.
        augmentation (bool, optional): Whether conventional (transformational) image augmentations are to be used. Defaults to False.
        aug_ratio (float, optional): If augmentation, the proportion of augmented images to create. Must be = 1. Defaults to 2.
        aug_var (float, optional): If augmentation, the transform variability parameter to use, typically a value between 0 and 5 (0 for no augmentations). Defaults to 3.
        weight_decay (float, optional): Weight decay parameter for the optimiser. Defaults to 0.
        device_str (str, optional): Name of the device to be used for model training. Defaults to "cpu".
        verbose (bool, optional): Whether updates should be given during training process. Defaults to True.
        CNN_params_dict (dict, optional): Can be helpful in other code to input all of these parameters in a single dict. If provided this will overwrite the other parameters. 
                                          Parameters included are epochs, batch_size, learning_rate, val_split, weight_decay and device. Defaults to None.
    Returns:
        model (PyTorch model): Final trained neural network model.
        val_score (float): Final validation score from the training, to use for hyperparameter optimisation.
        test_score (float): Test score from the training.
    """
    from torch.optim import Adam
    from torch.nn import NLLLoss
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    if CNN_params_dict is not None: # Allows to input all the parameters as a single dict overwritting other inputs
        epochs = CNN_params_dict["epochs"];                 batch_size = CNN_params_dict["batch_size"]
        learning_rate = CNN_params_dict["learning_rate"];   val_split = CNN_params_dict["val_split"]
        weight_decay = CNN_params_dict["weight_decay"];     device = CNN_params_dict["device"]

    # Parameter Type Validation (after unpacking CNN params dict)
    assert isinstance(model, torch.nn.Module), "model must be an instance of torch.nn.Module"
    assert isinstance(train_data, torch.utils.data.Dataset) or isinstance(train_data, list), "train_data must be an instance of torch.utils.data.Dataset"
    assert test_data is None or isinstance(test_data, torch.utils.data.Dataset) or isinstance(test_data, list), "test_data must be an instance of torch.utils.data.Dataset"
    assert isinstance(val_split, (int, float)) and 0 <= val_split < 1, "val_split must be a float between 0 and 1"
    assert isinstance(val_each_step, bool), "val_each_step must be a boolean"
    assert isinstance(epochs, int) and epochs > 0, "epochs must be a positive integer"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
    assert isinstance(learning_rate, float) and learning_rate > 0, "learning_rate must be a positive float"
    assert lr_sched is None or isinstance(lr_sched, str), "lr_sched must be a string or None"
    assert isinstance(augmentation, bool), "augmentation must be a boolean"
    if augmentation:
        assert isinstance(aug_ratio, (int, float, np.integer, np.floating)) and aug_ratio > 1, "aug_ratio must be a float greater than 1"
        assert isinstance(aug_var, (int, float)) and aug_var >= 0, "aug_var must be a float between 0 and 5 (can be above 5, but will be v messy)"
    assert isinstance(weight_decay, (int, float)), "weight_decay must be a numeric value"
    assert device_str in ["mps", "cuda", "cpu"], "device_str must be 'mps', 'cuda', or 'cpu'"
    assert isinstance(verbose, bool), "verbose must be a boolean"
    assert CNN_params_dict is None or isinstance(CNN_params_dict, dict), "CNN_params_dict must be a dictionary"

    if device_str == "mps" and torch.backends.mps.is_available(): device = torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available(): device = torch.device("cuda")
    elif device_str != "cpu":
        device = torch.device("cpu")
        warnings.warn(f"{device_str} device specified but not available, so using cpu")
    else: device = torch.device("cpu")
    model = model.to(device)

    #Setup training/validation data split
    train_size = int(len(train_data) * (1 - val_split))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])


    if augmentation: # Create new training data from image augmentation. Function uses img_transformation
        train_data = conv_augmenter(train_data, aug_ratio, aug_var)

    # Setup dataloaders
    train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    train_steps = max(len(train_data_loader.dataset) // batch_size, 1)
    if val_split > 0:
        val_data_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True)
        val_steps = max(len(val_data_loader.dataset) // batch_size, 1)

    # initialize a dictionary to store training history
    H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # optimiser and loss functions
    if opt_type == "adam": opt = Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    # elif: # add more types here as required
    else: raise ValueError("Optimiser type not recognised")

    if loss_type == "nll": loss_fn = NLLLoss()
    # elif: # add more types here as required
    else: raise ValueError("Loss function not recognised")
    
    if lr_sched is not None:
        lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor = 0.5, patience = 50, verbose = False)
    else: # Can add in options to use other types of lr_scheduler, not convinced on this one...
        lr_scheduler = None

    for e in range(epochs):
        # Training step
        model, total_train_loss, train_accuracy = nn_model_train_step(model, train_data_loader, device, loss_fn, opt)
        # Update training history
        avg_train_loss = total_train_loss / train_steps
        H["train_loss"].append(avg_train_loss)
        H["train_acc"].append(train_accuracy)
        if (val_split > 0 and val_each_step):
            # Validation step, if we have a non-zero validation set, and have specified to calculate it at each step
            model, total_val_loss, val_accuracy = nn_model_eval_step(model, val_data_loader, device, loss_fn)
            avg_val_loss = total_val_loss / val_steps
            H["val_loss"].append(avg_val_loss)
            H["val_acc"].append(val_accuracy)
        # print the model training and validation information
        if verbose:
            print("[INFO] EPOCH: {}/{}, learning rate = {}".format(e + 1, epochs, opt.param_groups[0]['lr']))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(total_train_loss / train_steps, train_accuracy))
            (print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avg_val_loss, val_accuracy)) if (val_split > 0 and val_each_step) else print(""))
            
        if lr_scheduler is not None:
            lr_scheduler.step(avg_val_loss)
            
    if (not val_each_step) and (val_split > 0): # If only calculating validation score at the end, do so here
        model, total_val_loss, val_accuracy = nn_model_eval_step(model, val_data_loader, device, loss_fn)

    # set some kind of score to output to compare across models
    val_score = (val_accuracy if val_split > 0 else None)
    # Now the model is trained, time to test it 
    
    # Set up test dataloader
    test_data_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # initialize a list to store our predictions
        y_pred = []; y_true = []
        # loop over the test set
        for (x, y) in test_data_loader:
            # send the input to the device
            x = x.to(device)
            # make the predictions and add them to the list
            pred = model(x)
            y_pred.extend(pred.max(1).indices.cpu().numpy())
            y_true.extend(y.cpu().numpy())
    # Get test data scores
    accuracy = accuracy_score(y_true, y_pred)
    test_score = accuracy
        
    return model, val_score, test_score

def num_from_syn_name(set_name):
    """ Function to get the replicate number from a synthetic set name. """
    match = re.search(r'\d+', set_name)
    if match:
        number = int(match.group())
    else:
        number = 1
        warnings.warn("No replicate number provided in the synthetic set name, so assuming 1.")
    return number

def synthetic_classifier_accuracy(synthetic_datasets, synthetic_label_sets, test_dataset, CNN_params, syn_augment = False, 
                                  x_train_real = None, y_train_real = None, augmentation = False, aug_var = 3, aug_ratio = 2, syn_vary_aug_ratio = False, device_str = "cpu"):
    """Assessing performance of synthetic data by how well it can be used to train a classifier. 
        Calculates the accuracy of a synthetic image set when used to train a classifier on CIFAR-10 data.
        Either training the classifier using the synthetic data only, or when used to augment a real dataset.
        Requires synthetic data & labels, test data & labels (the real data),
        parameters for the CIFAR CNN classifier model and optionally whether to augment real dataset (if provided).
    Args:
        synthetic_datasets (dict): The synthetic images to assess classifier accuracy on. Dict of datasets keyed by synthetic image type, each element is n x 3 x 32 x 32 array of n images.
        synthetic_label_sets (dict): Labels corresponding to the synthetic images. Dict of n-length lists, keyed by synthetic image type.
        test_dataset (TensorDataset): Real images to test the classifier on. Previous functions return a TensorDataset of CIFAR images to test on.
        CNN_params (dict): Input parameters for the CNN classifier model
        syn_augment (bool, optional): Whether to use the synthetic data to syn_augment a real dataset (if provided) for model training. Defaults to False.
        x_train_real (Tensor, optional): If syn_augment==True, the real training images to be syn_augmented with synthetic data. Tensor of m images (m x 3 x 32 x 32). Defaults to None.
        y_train_real (list, optional): If syn_augment==true, labels corresponding to the real training images. List of m images. Defaults to None.
        augmentation (Boolean, optional): Whether or not to carry out conventional data augmentations as part of the nn training. Defaults to False.
        aug_var (Float, optional): If augmentation==True, the variability parameter for the augmentations. Must be non-negative, typically <= 5. Defaults to 3.
        aug_ratio (Float, optional): If augment==True, the amount of data to create with conventional augmentations. Must be >1, defaults to 2.
        syn_vary_aug_ratio (Boolean, optional): If augment==True, this controls whether to vary the aug_ratio for synthetic sets to use less augmenting data with more synthetic replicates. Defaults to False.
        device_str (str, optional): Name of the device to be used for model training. Defaults to "cpu".
    Returns:
        dict: Classifier accuracies for each synthetic image type.
    """
    # Preset the device to use
    device = CNN_params["device"]
    # Preset a dictionary for the outputs
    accuracy_dict = {}
    for set_name, data in synthetic_datasets.items(): # loop through the input synthetic datasets
        if data is not None: #only carry on if there is data
            # Sort out the labels and x & y data for the given data subset
            labels = synthetic_label_sets[set_name]
            x_train = torch.from_numpy(np.array(data).reshape(-1, 3, 32, 32)).float()
            y_train = torch.tensor(labels).type(torch.LongTensor)
            
            if syn_augment: #If we want to use the synthetic data to syn_augment a real dataset, concatenate the two sets together
                # save the datasets as a tensor object, and check the values
                x_train_aug = torch.cat((x_train_real, x_train))
                y_train_aug = torch.cat((torch.tensor(y_train_real), y_train))
                syn_train_dataset = TensorDataset(x_train_aug, y_train_aug)
                assert ((x_train_aug >= 0).all() and (x_train_aug <= 1).all()), f"image data needs to be in range [0, 1] for array `x_train_aug` for synthetic set {set_name}"
            else:
                syn_train_dataset = TensorDataset(x_train, y_train)
                assert ((x_train >= 0).all() and (x_train <= 1).all()), f"image data needs to be in range [0, 1] for array `x_train` for synthetic set {set_name}"

            # If reducing the amount of augmented data for synthetic sets, use the num_from_syn_name function, and divide aug_ratio by that number.
            # If this is less than 1, warn the user and set the value to just over 1, otherwise the trainer will break.
            if syn_vary_aug_ratio:
                rep_count = num_from_syn_name(set_name)
                this_aug_ratio = aug_ratio / (1 + rep_count) # add 1 because the ratio includes the real data
                if this_aug_ratio < 1:
                    warnings.warn(f"""User has selected to adjust the augmentation ratio for different synthetic sets. 
                                  However, for set {set_name}, this would lead to a new aug_ratio less than or equal to 1, so setting to 1.001 instead. 
                                  Consider a larger value for aug_ratio.""")
                    this_aug_ratio = 1.001
            else:
                this_aug_ratio = aug_ratio

            # Load and train the classifier model on the (syn_augmented) synthetic training dataset, then and return calculate its accuracy on the test set
            # Define the model. We need to do this in each loop, as python stores it as a mutable object
            dropout_conv = CNN_params["dropout_conv"]
            dropout_fc = CNN_params["dropout_fc"]
            model = Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)

            _, _, test_accuracy = nn_trainer(model, syn_train_dataset, test_dataset, verbose = False, CNN_params_dict=CNN_params, 
                                                 augmentation=augmentation, aug_var=aug_var, aug_ratio=this_aug_ratio, device_str=device_str)

            accuracy_dict[set_name] = test_accuracy
    return accuracy_dict

def classifier_accuracies(synthetic_datasets, synthetic_label_sets, train_samp, test_dataset, syn_augment, images_true, y_train_true, device, device_name, 
                          augment = False, aug_var = 3, aug_ratio = 2, syn_vary_aug_ratio = False, verbose = True):
    """Function to calculate the classification-accuracy of a given synthetic dataset, compared against a given real set (assumed to be the generating set)
    Args:
        synthetic_datasets (dict): The synthetic images to assess classifier accuracy on. Dict of datasets keyed by synthetic image type, each element is n x 3 x 32 x 32 array of n images.
        synthetic_label_sets (dict): Labels corresponding to the synthetic images. Dict of n-length lists, keyed by synthetic image type.
        train_samp (dataset): Sample of real data to calculate baseline test score
        test_dataset (dataset): Holdout real test data
        syn_augment (Bool): Whether or not to syn_augment the synthetic data with the real data (provided below), as opposed to simply running on the synthetic-only.
        images_true (tensor): Real images to syn_augment the synthetic data with (if desired)
        y_train_true (list): Real labels accompanying the real images
        device (torch device): Device to run computing on
        device_name (str): Name of the above device
        augment (Boolean, optional): Whether or not to carry out conventional data augments as part of the nn training. Defaults to False.
        aug_var (Float, optional): If augment==True, the variability parameter for the augments. Must be non-negative, typically <= 5. Defaults to 3.
        aug_ratio (Float, optional): If augment==True, the amount of data to create with conventional augmentations. Must be >1, defaults to 2.
        syn_vary_aug_ratio (Boolean, optional): If augment==True, this controls whether to vary the aug_ratio for synthetic sets to use less augmenting data with more synthetic replicates. Defaults to False.
        verbose (bool, optional): Use this to control the amount of printed output. Defaults to True.
    Returns:
        CNN_params (dict): Set of parameters for the given CNN model
        baseline_test_acc (float): Output test accuracy for classifier trained on train_samp
        accuracy_dict (dict): Dict of output test accuracies for each synthetic datasets
    """
    # Get the parameters to use for the given CNN model and call the CNN model defined in classification_fns

    CNN_params = CNN_params_setup(device)
    dropout_conv = CNN_params["dropout_conv"]; dropout_fc = CNN_params["dropout_fc"]
    cnn_model = Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)

    ### ARE WE? For now, doubling the baseline epochs, to ensure that the baseline is stable
    baseline_CNN_params = CNN_params
    # baseline_CNN_params["epochs"] = baseline_CNN_params["epochs"] * 2

    # Start by calculating the test accuracy of the classifier when trained on the sample of real training images
    if verbose:
        print("Progress :: For validation by classifier accuracy, calculating baseline accuracy on the (real) test set.")
    _, _, baseline_test_acc  = nn_trainer(cnn_model, train_samp, test_dataset, device_str = device_name, 
                                                verbose = False, CNN_params_dict=baseline_CNN_params,
                                                augmentation=augment, aug_var=aug_var, aug_ratio=aug_ratio)
    if verbose:
        print(f"The test accuracy score for the baseline set is", round(baseline_test_acc, 4))
    # Then calculate the test accuracy of the classifier trained on each synthetic dataset
    # option of to whther these sets are syn_augmented with the real images or not
        print("Progress :: Calculating accuracies on the synthetic dataset "+("with" if syn_augment else "without") + " syn_augmenting real data.")
    accuracy_dict = synthetic_classifier_accuracy(synthetic_datasets, synthetic_label_sets, test_dataset, CNN_params, 
                                            syn_augment = syn_augment, x_train_real = images_true, y_train_real = y_train_true,
                                            augmentation=augment, aug_var=aug_var, aug_ratio=aug_ratio, syn_vary_aug_ratio=syn_vary_aug_ratio, device_str = device_name)
    return CNN_params, baseline_test_acc, accuracy_dict

def igpt_model_setup(device_name, pretrained_ref = 'openai/imagegpt-small'):
    # Open the ImageGPT model from torchvision, save it and set up CPU/GPU device
    feature_extractor = ImageGPTImageProcessor.from_pretrained(pretrained_ref)
    model = ImageGPTForCausalImageModeling.from_pretrained(pretrained_ref)
    clusters = feature_extractor.clusters
    device = torch.device(device_name)
    model.to(device)
    return feature_extractor, model, clusters, device

def img_decoder(data, clusters, img_sz = 32, prompt_frac = 1): 
    """ Code to convert convert color clusters in range (0, 1) back to pixels in range (0, 255) """
    if type(img_sz) == int: # Assume image is square
        img_sz_y = img_sz
        img_sz_x = img_sz
    elif len(img_sz) == 2: # Assume rectangular
        img_sz_y = img_sz[0]
        img_sz_x = img_sz[1]
    imgs = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [int(prompt_frac * img_sz_y),img_sz_x, 3]).astype(np.uint8) for s in data]
    return imgs

def image_plotter(images, rep_size, dpi = 180):
    """plot image(s) generated by the model
    Args:
        images (array): array of images to be plotted
        rep_size (int): rep size used for number of subplots
        dpi (int, optional): dpi for the plotting. Defaults to 180.
    """
    f, axes = plt.subplots(1, rep_size, dpi = dpi)
    for img,ax in zip(images, axes):
        ax.axis('off')
        ax.imshow(img)
    plt.show()

def image_saver(new_samples_img, out_path, prompt_sz_name, rep_size, seed, label):
    for img in new_samples_img:
        if not img.shape == (32, 32, 3):
            raise ValueError("Image not an array of expected shape for saving as png") # Check that images have been resized properly
        img_to_save = img.astype('uint8')
        rand_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 8)) #Set up a random 8-char string to ensure no overwriting issues
        # Save up a file name with the specified path, and the metadata for the image (prompt size, rep size and true label for classification)
        full_path = os.path.join(out_path, f'{prompt_sz_name}-prompt_rep-size_{rep_size}_seed_{seed}_{rand_name}_label-{label}.png') 
        cv2.imwrite(full_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)) #for some reason opencv uses BGR...
    print("Saved images to", out_path)

def create_igpt_img(real_img, img_sz, feature_extractor, clusters, model, device, rep_sizes = [1], prompt_sizes = ["med"], igpt_temp = 1, return_img_dict = False, save_images = False, out_path = None, label = None, seed = None, demo = False, verbose = False, gen_seed = None):
    # ensure that the model is correctly handling the image size
    if gen_seed is not None:
        np.random.seed(gen_seed)
        torch.manual_seed(gen_seed)
        random.seed(gen_seed)
        torch.cuda.manual_seed_all(gen_seed)

    if type(img_sz) == int: # Assume image is square
        img_sz_y = img_sz
        img_sz_x = img_sz
    elif len(img_sz) == 2: # Otherwise rectangular
        img_sz_y = img_sz[0]
        img_sz_x = img_sz[1]
    else:
        raise ValueError(f"Unrecognised output image size {img_sz}")
    
    img_dict = {}
    for rep_size in rep_sizes: # Loop through the rep sizes
        # Use the feature extractor to encode the pixel values, and create {rep_size} number of replicates
        # The function either accepts pixel values between -1 and 1, or normalises values between 0 and 255, so as our values are between 0 and 1, multiply by 255, then use the normaliser
        encoding = feature_extractor([(real_img * 255).to(torch.int)] * rep_size, return_tensors = "pt", do_normalize = True)
        encoded_raw_samples = encoding.input_ids.numpy()
        for i in range(len(prompt_sizes)): # Loop through the desired prompt sizes for each image now that they are encoded
            prompt_size = list(prompt_sizes.values())[i]
            prompt_sz_name = list(prompt_sizes.keys())[i]
            primers = encoded_raw_samples[:, :int(prompt_size * img_sz_y)*img_sz_x] # Create primers for the selected prompt size

            if demo: # If we want to plot images as the output, plot the raw and primer images for comparison
                # Decode the raw samples and the primers so that they can be plotted
                encoded_raw_samples_img = img_decoder(encoded_raw_samples, clusters, img_sz = img_sz)
                primers_img = img_decoder(primers, clusters, prompt_frac = prompt_size, img_sz = img_sz) # subset to adjust the plot size to the primer size
                if i == 0: # Only plot the full image for the first prompt size to minimise unneccessary clutter
                    image_plotter(encoded_raw_samples_img, rep_size)
                image_plotter(primers_img, rep_size)

            # Now for the actual snythetic images generation
            # Set up an image context from the set of primers for the given model size
            input_context = np.concatenate((np.full((rep_size, 1), model.config.vocab_size - 1), primers), axis=1)
            input_context = torch.tensor(input_context).to(device).to(torch.int)
            # Then use model.generate to create full sized images for the given prompt context
            synth_img = model.generate(input_ids=input_context, max_length=img_sz_y*img_sz_x + 1, temperature = igpt_temp, do_sample=True, top_k=40) 

            # Select the synthethic images and decode for full use
            new_samples = synth_img[:,1:].cpu().detach().numpy()
            new_samples_img = img_decoder(new_samples, clusters, img_sz = img_sz)

            if save_images: # Save the images with required names
                if out_path == None or label == None or seed == None:
                    raise ValueError("out_path, label and seed need to be specified for saving for valid filename")
                image_saver(new_samples_img, out_path, prompt_sz_name, rep_size, seed, label)
            if demo: # Demo plots
                image_plotter(new_samples_img, rep_size)
            for i, img in enumerate(new_samples_img):
                new_samples_img[i] = np.transpose(img, (2, 0, 1))
            img_dict[f"{prompt_sz_name}-{rep_size}"] = new_samples_img

    if return_img_dict: # Return an array of the images for further use
        return img_dict

def zipped_image_generator(zip_file_path, file_names):
    """ Function to open images out of a zipped file, using yield to give a generator (allowing to open without loading all images to memory)"""
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in file_names:
            with zip_ref.open(file_name) as file_data:
                image_data = np.frombuffer(file_data.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                yield os.path.basename(file_name), image

def synthetic_type_split_import(folder_path, split_method, name_cond = None, zipped = False, prompt_set = "full"):
    """Function to support the ImageGPT synthetic data import. 
    If desired to split the data by type, then this function is run. 
    Args:
        folder_path (str): Filepath (relative to current file) of the folder of synthetic images to import
        split_method (str): E.g. "prompt_rep", the method with which to split the data (prompt_rep does this by prompt size name and rep size number)
        name_cond (str or None, optional): Whether to only select images whose name select a certain string. Set to None if no such filtering desired. Defaults to None.
        zipped (bool, optional): If the selected folder is zipped, need to additionally unzip in the process. Defaults to False.
        prompt_set (str or list, optional): The set of prompts to use. Either "full" for the default 12: s/m/l 2/4/6/8, or specify a list. Defaults to "full"
    Returns:
        synthetic_datasets, synthetic_label_sets: dictionaries of the split datasets and label_sets, keyed by the split method (e.g. prompt & rep sizes)
    """
    if split_method == "prompt_rep":
        # To sort them by prompt and rep size for training the different classifiers, 
        # make a dict of 12 synthetic datasets (and 12 label sets) for all prompt size & rep size combos
        if prompt_set == "full":
            prompt_sizes = ['small', 'med', 'large']
            rep_sizes = ['2', '4', '6', '8']
        else:
            prompt_sizes = list(set(item.split('-')[0] for item in prompt_set))
            rep_sizes = list(set(item.split('-')[1] for item in prompt_set))
            assert set(prompt_sizes).issubset(['small', 'med', 'large']), "Invalid prompts found"
            assert all(rep.isdigit() for rep in rep_sizes), "Invalid reps found"

        synthetic_datasets = {}; synthetic_label_sets = {}
        for prompt in prompt_sizes:
            for rep in rep_sizes:
                synthetic_datasets[prompt + '-' + rep] = None
                synthetic_label_sets[prompt + '-' + rep] = []
        if zipped:
            with zipfile.ZipFile(folder_path, 'r') as zip_ref:
                file_names = sorted(zip_ref.namelist())
                skipped_elts = file_names[0:1]
                if any ('.png' in element for element in skipped_elts):
                    raise ValueError("Accidentally deleted synthetic image while unzipping folder.")
                file_names = file_names[1:]
            for filename, image in zipped_image_generator(folder_path, file_names):
                if filename.endswith('.png'): # Check is a png
                    if name_cond is None or name_cond in filename: # If no name_cond filter is specified, or if the desired string IS in the current filename, continue
                        if image is not None: # save the image
                            # Get the prompt & rep sizes from the image title
                            prompt = filename[:filename.find('-prompt')]
                            rep = filename[filename.find('rep-size_')+len('rep-size_')]
                        # Initialize a 4D array if it's not already initialized to store the data in
                            if synthetic_datasets[prompt + '-' + rep] is None:
                                synthetic_datasets[prompt + '-' + rep] = np.transpose(image, (2, 0, 1))[np.newaxis, ...]  # reshape the image to 3 x 32 x 32 as that's how the classifier is set up and add to 4D array
                            else:
                                # Append the current image to the 4D array
                                synthetic_datasets[prompt + '-' + rep] = np.vstack((synthetic_datasets[prompt + '-' + rep], np.transpose(image, (2, 0, 1))[np.newaxis, ...]))
                            label_index = filename.find('label-') # And save the corresponding label for classification
                            if label_index != -1:
                                label_value = int(filename[label_index+len('label-')])
                                synthetic_label_sets[prompt + '-' + rep].append(label_value)
                            else:
                                raise ValueError(f"No label found for image {filename}")
                            
        else:
            for filename in os.listdir(folder_path): # Loop through all files in the folder
                if filename.endswith('.png'): # Check is a png
                    if name_cond is None or name_cond in filename: # If no name_cond filter is specified, or if the desired string IS in the current filename, continue
                        image_path = os.path.join(folder_path, filename)
                        image = cv2.imread(image_path) # Read it as an image (reads to n x n x 3 BGR numpy array)
                        if image is not None: # save the image
                            # Get the prompt & rep sizes from the image title
                            prompt = filename[:filename.find('-prompt')]
                            rep = filename[filename.find('rep-size_')+len('rep-size_')]

                        # Initialize a 4D array if it's not already initialized to store the data in
                            if synthetic_datasets[prompt + '-' + rep] is None:
                                synthetic_datasets[prompt + '-' + rep] = np.transpose(image, (2, 0, 1))[np.newaxis, ...]  # reshape the image to 3 x 32 x 32 as that's how the classifier is set up and add to 4D array
                            else:
                                # Append the current image to the 4D array
                                synthetic_datasets[prompt + '-' + rep] = np.vstack((synthetic_datasets[prompt + '-' + rep], np.transpose(image, (2, 0, 1))[np.newaxis, ...]))
                            label_index = filename.find('label-') # And save the corresponding label for classification
                            if label_index != -1:
                                label_value = int(filename[label_index+len('label-')])
                                synthetic_label_sets[prompt + '-' + rep].append(label_value)
                            else:
                                raise ValueError(f"No label found for image {filename}")
                    
        # save the images as floats between 0 and 1
        for key, value in synthetic_datasets.items():
            if value is not None: # in case the dict is empty
                new_val = value/255
                synthetic_datasets[key] = new_val
                assert ((new_val >= 0).all() and (new_val <= 1).all()), f"image data needs to be in range [0, 1] for array `new_val` in synthetic dataset {key}"

        return synthetic_datasets, synthetic_label_sets
    else:
        raise ValueError(f"Unknown method to categorise/split the synthetic data provided by `split_method`={split_method}")
    
def synthetic_data_import(folder_path, zipped = False, split_by_type = False, split_method = "prompt_rep", seed_subset = False, prompt_set = "full"):
    """Function to import the synthetic images from a given folder, and split by type or seed if desired to create different datasets.
    Args:
        folder_path (str): Filepath (relative to current file) of the folder of synthetic images to import
        zipped (bool): If the selected folder is zipped, need to additionally unzip in the process. Defaults to False.
        split_by_type (str): Whether to split the output data by type (as specified in split_method). Defaults to False.
        split_method (str): The method with which to split the data (prompt_rep does this by prompt size name and rep size number). Defaults to \'prompt_rep\'".
        seed_subset (bool, optional): If desired to also output the data as split by the seed in the image name. Defaults to False.
        prompt_set (str or list, optional): The set of prompts to use. Either "full" for the default 12: s/m/l 2/4/6/8, or specify a list. Defaults to "full"
    Returns:
        if split_by_type: 
            if seed_subset: synthetic_datasets_by_seed, synthetic_label_sets_by_seed, each dictionaries keyed by the seed values in the image names. 
                            Each element of the dictionary is a dictionary of datasets, keyed by the split method (e.g. prompt & rep sizes)
            else not seed_subset: synthetic_datasets, synthetic_label_sets, dictionaries of datasets keyed by the split method (e.g. prompt & rep sizes)
        else not split_by_type: raw_syn_data, a 4D tensor of the images (e.g. n x 32 x 32 x 3 for the CIFAR data), and labels, a 1D list of correpsonding labels
    """
    if split_by_type: # this will separate the images out by prompt and replication size
        if seed_subset:
            if zipped:
                file_list = zipfile.ZipFile(folder_path, 'r').namelist()
                skipped_elts = file_list[0:1]
                if any ('.png' in element for element in skipped_elts):
                    raise ValueError("Accidentally deleted synthetic image while unzipping folder.")
                file_list = file_list[1:]
            else:
                file_list = os.listdir(folder_path)
            seed_values = set()
            for file_name in file_list:
                if "_seed_" in file_name:
                    start_index = file_name.find("_seed_") + len("_seed_")
                    end_index = file_name.find("_", start_index)
                    seed_value = file_name[start_index:end_index]
                    seed_values.add(seed_value)
                else:
                    raise ValueError(f"Unexpected name \"{file_name}\" for image in {folder_path}, was expecting to include\"_seed_\"")
            seed_values = sorted(seed_values)

            synthetic_datasets_by_seed = {}; synthetic_label_sets_by_seed = {}
            for seed in seed_values:
                synthetic_result = synthetic_type_split_import(folder_path, split_method, name_cond = "_seed_"+seed, zipped = zipped)
                synthetic_datasets_by_seed[seed] = synthetic_result[0]
                synthetic_label_sets_by_seed[seed] = synthetic_result[1]
            return synthetic_datasets_by_seed, synthetic_label_sets_by_seed

        else:
            synthetic_datasets, synthetic_label_sets = synthetic_type_split_import(folder_path, split_method, zipped = zipped, prompt_set = prompt_set)
            return synthetic_datasets, synthetic_label_sets

    else: # Otherwise, instead of splitting by prompt & rep size, just load in as a 4D tensor
        if seed_subset:
            raise NotImplementedError("I haven't yet coded doing subsetting by seed for the non-type-split data output.")
        raw_syn_data = []
        labels = []
        for filename in os.listdir(folder_path): # Loop through all files in the folder
            if filename.endswith('.png'): # Check is a png
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path) # Read it as an image (reads to n x n x 3 BGR numpy array)
                raw_syn_data.append(image)
            
            # Get the label
            label_index = filename.find('label-')
            if label_index != -1:
                label = int(filename[label_index+len('label-')])
            else:
                raise ValueError(f"No label found for image {filename}")
            labels += [label]

        # save the images as floats between 0 and 1
        for i in range(len(raw_syn_data)):
            image = raw_syn_data[i]/255
            assert ((image >= 0).all() and (image <= 1).all()), f"image data needs to be in range [0, 1] for array `new_val` in synthetic dataset"
            raw_syn_data[i] = image
        
        # Save as a 4D torch tensor of images as 1D list of labels
        raw_syn_data = np.stack(raw_syn_data, axis = 0)
        raw_syn_data = torch.from_numpy(raw_syn_data).to(dtype=torch.float32)
        raw_syn_data = raw_syn_data.permute(0, 3, 1, 2) # ensures consistent data structure
        return raw_syn_data, labels

def scores_compile(synthetic_datasets, scores_dict, baseline_test_acc = None):
    """Function to compile synthetic scores from the different val_methods into one score_df
    Args:
        synthetic_datasets (dict): The synthetic images to assess classifier accuracy on. Dict of datasets keyed by synthetic image type, each element is n x 3 x 32 x 32 array of n images.
        scores_dict (dict): Validation scores for each of the methods above
        baseline_test_acc (Float, optional): Baseline test accuracy, required if class_acc is in val_methods. Defaults to None.
    Returns:
        DataFrame: df with columns of scores for each validation method and some descriptors for each synthetic set
    """
    # Start by setting up descriptors for the synthetic sets, and converting them to useful names and values for output
    scores_list = []
    warnings.warn("Currently assumping that the synthetic sets are described with prompt names (small/med/large) and rep sizes (2/4/6/8)")
    scores_col_names = ["set_name", "prompt_val", "prompt_name", "reps"]
    # Loop through the sets and add each set's descriptors and scores
    for syn_set in synthetic_datasets.keys():
        # For each synthetic set, get it's set-type descriptors
        prompt = syn_set[:syn_set.find('-')]
        if prompt in ['small', 'med', 'large']: #convert small/med/large prompt to 0.25/0.5/0.75 and quarter/half/three-quarters for the plotting
            prompt_val = {'small': 0.25, 'med': 0.5, 'large': 0.75}[prompt]
            prompt_name = {'small': 'quarter', 'med': 'half', 'large': 'three-quarters'}[prompt]
        else:
            raise ValueError(f"Unrecognised prompt name, {prompt}")
        reps = syn_set[syn_set.find('-')+1:]
        set_score_list = [syn_set, prompt_val, prompt_name, reps]
        # Then for each validation method, add that methods' score to the score list, and check its name is in the columns list
        val_methods = ["class_acc", "fid_inf"]
        for val_method in val_methods:
            set_score_list += [scores_dict[val_method][syn_set]]
            if val_method not in scores_col_names:
                scores_col_names += [val_method]
            if val_method == "class_acc": # If we're testing classification accuracy, also compute the accuracy index from the baseline_test_acc
                acc = scores_dict[val_method][syn_set]
                index = acc / baseline_test_acc # * 100
                set_score_list += [index]
                if "acc_ind" not in scores_col_names:
                    scores_col_names += ["acc_ind"]
        scores_list.append(set_score_list)
    # With a list of lists of scores and corresponding column names, convert these into a dataframe
    score_df = pd.DataFrame(scores_list, columns = scores_col_names)
    return score_df

def seed_subset_accuracies(seed, syn_data_dict, synthetic_label_sets_by_seed, train_dataset, n_samp, test_dataset, syn_augment, 
                           images_true, y_train_true, device, device_name, augment = False, aug_var = 3, aug_ratio = 2, syn_vary_aug_ratio = False):
    """ Function to calculate classifier accuracy on seeded set of synthetic datasets (multiple synthetic datasets of different types generated from one seed).
        Calculate the corresponding training sample for that seed (if needed for syn_augmentation), and calculate the classifier accuracy dict for given set of synthetic datasets. """
    seed_label_set = synthetic_label_sets_by_seed[seed]    
    # set up the real data to be syn_augmented with (if desired) - make sure to sample with the same seed as the given synthetic set for most representative results
    train_samp, images_true, y_train_true = train_sampler(train_dataset, n_samp, seed = int(seed))

    _, seed_baseline_test_acc, seed_accuracy_dict = classifier_accuracies(syn_data_dict, seed_label_set, train_samp, test_dataset, syn_augment, images_true, y_train_true, device, device_name, 
                                                                         augment = augment, aug_var = aug_var, aug_ratio = aug_ratio, syn_vary_aug_ratio = syn_vary_aug_ratio, verbose = False)
    return seed_baseline_test_acc, seed_accuracy_dict    

def calc_err_bars(score_df, other_seeds_path, train_dataset, n_samp, test_dataset, CNN_params, syn_augment, 
                       images_true, y_train_true, baseline_test_acc, device, device_name,
                       zipped = False, augment = False, aug_var = 3, aug_ratio = 2, syn_vary_aug_ratio = False):
    """Function to extend the charts with error bars calculated across multiple sets of synthetic datasets.
        Assumes score_df contains one dataset from seed 42 (gives an extra row of data for no additional work.)
    Args:
        score_df (DataFrame): Dataframe of scores from existing experiments (minimises code repetition)
        other_seeds_path (string): filepath of folder to access the data from
        train_dataset (Dataset): Original training data (to be re-sampled from to equate with synthetic data)
        n_samp (int): Sample size 
        test_dataset (Dataset): Holdout test dataset to calculate accuracies against
        CNN_params (dict): Set of parameters for the given CNN model
        syn_augment (Bool): Whether or not to syn_augment the synthetic data with the real data (provided below)
        images_true (tensor): Real images to syn_augment the synthetic data with (if desired)
        y_train_true (List): Real labels accompanying the real images.
        baseline_test_acc (Float): Baseline test accuracy from existing experiments, for computing accuracy indexes (minimises code repetition).
        device (torch device): Device to run computing on
        device_name (str): Name of the above device
        zipped (bool): If the selected folder is zipped, need to additionally unzip in the process. Defaults to False.
        dstl_fid_err (bool): Whether to additionally calculate errors for the FID. Defaults to False.
        augment (Boolean, optional): Whether or not to carry out non-synthetic (transformational)data augments as part of the nn training. Defaults to False.
        aug_var (Float, optional): If augment==True, the variability parameter for the augments. Must be non-negative, typically <= 5. Defaults to 3.
        aug_ratio (Float, optional): If augment==True, the amount of data to create with non-synthetic augmentations. Must be >1, defaults to 2.
        syn_vary_aug_ratio (Boolean, optional): If augment==True, this controls whether to vary the aug_ratio for synthetic sets to use less augmenting data with more synthetic replicates. Defaults to False.
    Returns:
        score_df_combo (DataFrame): df of the different accuracy, accuracy index and FID (if specified) scores for each of the synthetic subsets.
        acc_index_df (DataFrame): df of the accuracies per index (useful for boxplot)
        experiment_count (int): The number of experiments run over/the number of repeat datasets analysed (useful as reference).
    """
    # Import the by-seed sets of synthetic datasets.
    synthetic_datasets_by_seed, synthetic_label_sets_by_seed = synthetic_data_import(other_seeds_path, split_by_type = True, split_method = "prompt_rep", seed_subset = True, zipped = zipped)
    print("Progress :: To plot accuracy error bars, computing classifier accuracies for other synthetic datasets.")
    accuracies_dict = {}
    baseline_accuracies_dict = {}
    fids_dict = {}
    for seed, syn_data_dict in synthetic_datasets_by_seed.items():
        print(f"Progress :: Computing accuracies for seed {seed} out of {list(synthetic_datasets_by_seed.keys())}")
        seed_baseline_test_acc, seed_accuracy_dict = seed_subset_accuracies(seed, syn_data_dict, synthetic_label_sets_by_seed, train_dataset, n_samp, test_dataset, syn_augment, 
                                                            images_true, y_train_true, device, device_name, augment=augment, 
                                                            aug_var=aug_var, aug_ratio=aug_ratio, syn_vary_aug_ratio=syn_vary_aug_ratio)

        accuracies_dict[seed] = seed_accuracy_dict
        baseline_accuracies_dict[seed] = seed_baseline_test_acc
        # If either FID method is specified in val_methods, and FID errors are desired, calculate the FID for each synthetic dataset.

        seed_fid_dict = calculate_fid_infs(syn_data_dict, images_true, CNN_params["device"], min_fake = 900)
        fids_dict[seed] = seed_fid_dict

    # Reshape the existing outputs for error bar plotting
    baseline_accuracies_dict["42"] = baseline_test_acc
    accuracies_dict["42"] = dict(zip(score_df["set_name"], score_df["class_acc"]))
    accuracies_df = pd.DataFrame.from_dict(accuracies_dict, orient="index")

    # Calculate the accuracy index from the baseline accuracies
    acc_index_df = accuracies_df.div(accuracies_df.index.map(baseline_accuracies_dict), axis=0)#.multiply(100)
    # Calculate the median, lower and upper quartile accuracies & acc index for the errorbar plots
    agg_results = {}
    for col in accuracies_df.columns:
        lq_acc, med_acc, uq_acc = np.percentile(accuracies_df[col], [25, 50, 75], axis = 0) # I think axis = 0 is right?
        lq_ind, med_ind, uq_ind = np.percentile(acc_index_df[col], [25, 50, 75], axis = 0)
        agg_results[col] = {
            'lq_acc': lq_acc, 'med_acc': med_acc, 'uq_acc': uq_acc,
            'lq_acc_ind': lq_ind, 'med_acc_ind': med_ind, 'uq_acc_ind': uq_ind}
    agg_acc_df = pd.DataFrame(agg_results).T
    agg_acc_df.index.name = "set_name"
    agg_acc_df = agg_acc_df.reset_index()
    # Merge these into the main score df, then calculate the error between median and upper/lower quartile accuracy index
    score_df_combo = pd.merge(score_df.drop(columns=["class_acc", "acc_ind"], inplace=False), agg_acc_df, on = "set_name")
    score_df_combo['uq_ind_err'] = score_df_combo['uq_acc_ind'] - score_df_combo['med_acc_ind']
    score_df_combo['lq_ind_err'] = score_df_combo['med_acc_ind'] - score_df_combo['lq_acc_ind']
    
    # If FID errors are specified, and the method is specified (regular or FID_inf), carry on with the second plot. If both methods are specified, prioritise FID_inf.
    fid_type = "fid_inf"
    # Similar to accuracies, add in the FID value from the extra dataset from score_df. 
    # Then calculate the median, uq & lq, merge these into the main score_df, and calculate the respective errors for errorbar plotting
    fids_dict["42"] = dict(zip(score_df["set_name"], score_df[fid_type]))
    fids_df = pd.DataFrame.from_dict(fids_dict, orient="index")
    agg_fid_results = {}
    for col in fids_df.columns:
        lq_fid, med_fid, uq_fid = np.percentile(fids_df[col], [25, 50, 75], axis = 0)
        agg_fid_results[col] = {
            'lq_fid': lq_fid, 'med_fid': med_fid, 'uq_fid': uq_fid}
    agg_fid_df = pd.DataFrame(agg_fid_results).T
    agg_fid_df.index.name = "set_name"
    agg_fid_df = agg_fid_df.reset_index()
    score_df_combo = pd.merge(score_df_combo, agg_fid_df, on = "set_name")
    score_df_combo["uq_fid_err"] = score_df_combo["uq_fid"]-score_df_combo["med_fid"]
    score_df_combo["lq_fid_err"] = score_df_combo["med_fid"]-score_df_combo["lq_fid"]

    experiment_count = len(accuracies_df)
    return score_df_combo, acc_index_df, experiment_count

def split_dataset_by_class(dataset, class_list):
    sorted_data = {key: [] for key in class_list}
    for image, label in dataset:
        sorted_data[int(label)] += [image]
    for key, images in sorted_data.items():
        sorted_data[key] = TensorDataset(torch.from_numpy(np.stack(images)))
    return sorted_data

def sample_split_datasets(split_datasets, n_samp, rand_seed = 42):
    sampled_datasets = {key: [] for key in split_datasets.keys()}
    for key, dataset in split_datasets.items():
        data_samp = data_sampler(dataset, n_samp, rand_seed = rand_seed)
        dataset_samp = TensorDataset(torch.from_numpy(np.stack([item[0] for item in data_samp])))
        sampled_datasets[key] = dataset_samp
    return sampled_datasets

def syn_dict_to_dataset(syn_imgs_dict, shuffle = True):
    imgs = []
    labels = []
    for label, label_imgs in syn_imgs_dict.items():
        n_syn_img = len(label_imgs)
        imgs += label_imgs
        labels += [label] * n_syn_img

    syn_dataset = TensorDataset(torch.stack(imgs), torch.tensor(labels)) # Note this is NOT shuffled
    if shuffle:
        shuf_indices = torch.randperm(len(syn_dataset))
        syn_dataset = Subset(syn_dataset, shuf_indices)
    return syn_dataset

# Set up the GAN architectures
def weights_init(neural_net):
    """ Initializes the weights of the neural network
    :param neural_net: (De-)Convolutional Neural Network where weights should be initialized
    """
    classname = neural_net.__class__.__name__
    if classname.find('Conv') != -1:
        neural_net.weight.data.normal_(0, 2e-2)
    elif classname.find('BatchNorm') != -1:
        neural_net.weight.data.normal_(1, 2e-3) # original was sd 2e-2, pytorch cgan recommended sd 0
        neural_net.bias.data.fill_(0)

class Generator_v2(nn.Module):
    """Generates artificial images form a random vector as input."""

    def __init__(self, input_size=100, n_feat=64, n_chan=3, n_layers=5):
        if n_layers < 4:
            raise ValueError("Cannot handle fewer than 4 CNN layers in the generator.")
        super(Generator_v2, self).__init__()

        layers = []
        # First convolutional layer
        out_ratio = pow(2, n_layers - 2)
        layers.append(nn.ConvTranspose2d(input_size, n_feat * out_ratio, 4, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(n_feat * out_ratio))
        layers.append(nn.ReLU(True))

        # Intermediate convolutional layers
        for i in range(n_layers - 2, 0, -1):
            in_ratio = pow(2, i)
            out_ratio = pow(2, i - 1)
            kernel_size = 4 if i <= 3 else 3  # Adjust kernel size based on output ratio
            stride = 2 if i <= 3 else 1  # Adjust stride based on output ratio
            layers.append(nn.ConvTranspose2d(n_feat * in_ratio, n_feat * out_ratio, kernel_size, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(n_feat * out_ratio))#, track_running_stats=False))
            layers.append(nn.ReLU(True))
        # Last convolutional layer
        final_kernel_size = 4 if n_layers <= 4 else 3  # Adjust kernel size based on the number of layers
        final_stride = 2 if n_layers <= 4 else 1  # Adjust stride based on the number of layers
        layers.append(nn.ConvTranspose2d(n_feat, n_chan, final_kernel_size, final_stride, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input_vector):
        return self.main(input_vector)

class Discriminator_v2(nn.Module):
    """Evaluates the artificial images from the Generator and
    either accepts or rejects the image (returns value between 0 - 1).
    """

    def __init__(self, n_feat=64, n_chan=3, p_drop=0, n_layers=5):
        if n_layers < 5:
            raise ValueError("Cannot handle fewer than 4 CNN layers in the discriminator.")
        super(Discriminator_v2, self).__init__()

        layers = []
        # First convolutional layer
        layers.append(nn.Conv2d(n_chan, n_feat, 4, 2, 1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Dropout(p_drop))
        # Intermediate convolutional layers
        for i in range(1, n_layers - 1):  # 1 until n-1 because we already have the first and last layer
            in_ratio = pow(2, i - 1)
            out_ratio = pow(2, i)
            kernel_size = 4 if i <= 4 else 3  # Adjust kernel size based on output ratio
            stride = 2 if i <= 4 else 1  # Adjust stride based on output ratio
            layers.append(nn.Conv2d(n_feat * in_ratio, n_feat * out_ratio, kernel_size, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(n_feat * out_ratio))#, track_running_stats=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p_drop))
        # Last convolutional layer
        in_ratio = pow(2, n_layers - 2)
        final_kernel_size = 4 if n_layers <= 5 else 3  # Adjust kernel size based on the number of layers
        final_stride = 2 if n_layers <= 5 else 1  # Adjust stride based on the number of layers
        layers.append(nn.Conv2d(n_feat * in_ratio, 1, final_kernel_size, final_stride, 1, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input_image):
        return self.main(input_image).view(-1)
    
def save_model(state_dict, name, epoch, path_prefix = "gan_training", verbose = False):
    """Saves the trained neural net, optimizer and epoch
    :param state_dict: Dictionary of states from the (De-)Convolutional Neural Network, optimizer & epoch
    :param name: Name of the Neural Network
    :param epoch: Current epoch
    """
    path = path_prefix + "/model"
    if verbose:
        print(f"        Saving trained model {name} at epoch {epoch}.")
    if not os.path.exists(path):
        os.makedirs(path)
    model_name = 'dcgan_{}_{}.pth'.format(name, epoch)
    model_path = os.path.join('./'+path, model_name)
    for key in state_dict.keys():
        if 'running_mean' in key or 'running_var' in key:
            del state_dict[key]
    torch.save(state_dict, model_path)

def save_images(imgs, name, epoch, path_prefix = "gan_training", verbose = False):
    """Saves images
    :param imgs: Images
    :param name: Name for images
    :param epoch: Current epoch
    """
    path = path_prefix + "/results"
    if verbose:
        print(f"        Progress ::Saving images {name} at epoch {epoch}.")
    if not os.path.exists(path):
        os.makedirs(path)
    img_name = '{}_{:03d}.png'.format(name, epoch)
    img_path = os.path.join('./'+path, img_name)
    vutils.save_image(imgs, img_path, normalize=True)
    
def load_model(filepath, neural_net, optimizer):
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath, strict=False)
        neural_net.load_state_dict(ckpt['state_dict'])

        # If BatchNorm statistics are saved in the checkpoint, load them
        if 'batch_norm_stats' in ckpt:
            for name, module in neural_net.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean = ckpt['batch_norm_stats'][name]['running_mean']
                    module.running_var = ckpt['batch_norm_stats'][name]['running_var']

        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Progress :: Loading trained model {neural_net.__class__} at epoch {ckpt['epoch']}")# and resuming training")
        return neural_net, optimizer, ckpt['epoch']

def train_gan(train_data, resume_generator = False, resume_discriminator = False, total_epochs = 25, verbose = 1, save_path = "gan_training", save_type = "final", 
              batch_size = 64, loader_num_workers = 2, val_split = 0, device_name = "cpu", lr_gen = 2e-4, lr_disc = 2e-4, opt_beta_1 = 0.5, opt_beta_2 = 0.999, weight_decay = 0,
              nn_input_size = 100, nn_n_feat = 64, nn_n_layers_gen = 5, nn_n_layers_disc = 5, nn_p_drop = 0, label_smoothing = 0, disc_noise_std = 0,
              augmentation = False, aug_ratio = 2, aug_var = 3, gan_type = "v2"):
    """Function for carrying out the training procedure of our GAN.

    Args:
        train_data (Torch Dataset): Original data which we are seeking to create synthetic data from
        resume_generator (False bool/path_str, optional): Whether we are resuming training; if so, provide the path to the generator model. Defaults to False.
        resume_discriminator (False bool/path_str, optional): Whether we are resuming training; if so, provide the path to the generator model. Defaults to False.
        total_epochs (int, optional): The total number of epochs to train for. Defaults to 25.
        verbose (float>=0, optional): The amount of information to print during training. Currently controlled by 0, 1 or 2. Defaults to 1.
        save_path (str, optional): Location to save models to. Defaults to "gan_training".
        save_type (str, optional): str from ["final", "all", "ten_intermediate", "none"], representing how many iterations of the model to save. Defaults to "final".
        batch_size (int, optional): Number of images per training batch. Defaults to 64.
        loader_num_workers (int, optional): Number of workers in the dataloader (not sure what this does...). Defaults to 2.
        val_split (0<=float<1, optional): The proportion of real images to hold out for validating the discriminator through the trainings runs. Defaults to 0.
        device_name (str, optional): Name of the device to carry out training on, from ["cpu", "mps", "cuda"]. Defaults to "cpu".
        lr_gen (0<=float<1, optional): Learning rate parameter for the generator's optimiser. Defaults to 2e-4.
        lr_disc (0<=float<1, optional): Learning rate parameter for the discriminator's optimiser. Defaults to 2e-4.
        opt_beta_1 (0<=float<=1, optional): beta_1 momentum parameter for the optimisers. Defaults to 0.5.
        opt_beta_2 (0<=float<=1, optional): beta_2 error variance parameter for the optimisers. Defaults to 0.999.
        weight_decay (0<=float<1, optional): weight decay parameter for the optimised. Defaults to 0.
        nn_input_size (int, optional): Size of input layer to the generator. Defaults to 100.
        nn_n_feat (int, optional): Number of features for the networks. Defaults to 64.
        nn_n_layers_gen (4<=int<=8, optional): Number of layers in the generator. Must be greater than 4, memory issues above 7 or 8. Defaults to 5.
        nn_n_layers_disc (5<=int<=8, optional): Number of layers in the disciminator. Must be greater than 5, memory issues above 7 or 8. Defaults to 5.
        nn_p_drop (0<=float<1, optional): Dropout probability for the discriminator. Defaults to 0.
        label_smoothing (0<float<0.5, optional): Parameter to smooth the discriminator labels to reduce overconfidence. Probably anything above 0.2 is risky. Defaults to 0.
        disc_noise_std (float>=0, optional). Standard deviation of noise to add to images to feed into discriminator to minimise vanishing gradient. Defaults to 0.
        augmentation (bool, optional): Whether conventional (transformational) image augmentations are to be used. Defaults to False.
        aug_ratio (float, optional): If augmentation, the proportion of augmented images to create. Must be >= 1. Defaults to 2.
        aug_var (float, optional): If augmentation, the transform variability parameter to use, typically a value between 0 and 5 (0 for no augmentations). Defaults to 3.
        gan_type (str, optional): Type of GAN model to train. Currently accepts one of ["v1", "v2", "cond_v1"]

    Returns:
        Saved models in specified path & type
        disc_state_dict: state_dict for the discriminator
        gen_state_dict: state_dict for the generator
        overall_mean_disc_loss: Average discriminator loss across batches and labels
        overall_mean_gen_loss: Average generator loss across batches and labels
        overall_mean_val_disc_loss: Average discriminator validation loss across batches and labels
    """

    if device_name == "mps" and torch.backends.mps.is_available(): 
        device = torch.device("mps")
        print(f"Training GAN on {device_name.upper()} device") if verbose > 0 else ""
    elif device_name == "cuda" and torch.cuda.is_available(): 
        device = torch.device("cuda")
        print(f"Training GAN on {device_name.upper()} device") if verbose > 0 else ""
    elif device_name != "cpu":
        device = torch.device("cpu")
        warnings.warn(f"{device_name} device specified but not available, so using CPU")
    else: 
        device = torch.device("cpu")
        print(f"Training GAN on CPU device") if verbose > 0 else ""

    cgan = False # set up a flag of if running a cgan (including labels in input)

    if gan_type == "v2":
        net_generator = Generator_v2(input_size = nn_input_size, n_feat = nn_n_feat, n_layers = nn_n_layers_gen).apply(weights_init)
        net_discriminator = Discriminator_v2(n_feat = nn_n_feat, n_layers = nn_n_layers_disc, p_drop = nn_p_drop).apply(weights_init)
    else:
        raise ValueError(f"Unrecognised gan_type input: {gan_type}")

    optimizer_generator = optim.Adam(net_generator.parameters(),
                                     lr=lr_gen, weight_decay=weight_decay,
                                     betas=(opt_beta_1, opt_beta_2))
    optimizer_discriminator = optim.Adam(net_discriminator.parameters(),
                                         lr=lr_disc, weight_decay=weight_decay,
                                         betas=(opt_beta_1, opt_beta_2))

    start_epoch = 0
    if resume_generator:
        net_generator, optimizer_generator, start_epoch = load_model(resume_generator,
                                                                     net_generator,
                                                                     optimizer_generator)
    if resume_discriminator:
        net_discriminator, optimizer_discriminator, start_epoch = load_model(resume_discriminator,
                                                                             net_discriminator,
                                                                             optimizer_discriminator)
    criterion = nn.BCELoss()

    # Send the models to the device to be used
    (net_generator, net_discriminator) = (net_generator.to(device), net_discriminator.to(device))

    ### Set up the data
    # If we're creating a holdout validation set, split it out here
    # then create dataloaders for the validation & training data
    if val_split >= 1:
        raise ValueError(f"Validation split ratio {val_split} needs to be less than 1.")
    elif val_split > 0:
        # simply split the data by the proportion set in val_split
        train_size = int(len(train_data) * (1 - val_split))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)

    # Augment the training data if desired using the classification_functions transformation augmenter.
    if augmentation: # Create new training data from image augmentation. Function uses img_transformation
        train_data = conv_augmenter(train_data, aug_ratio, aug_var, labelled_data=False, random_erase_greyscale=False) # Random erasures & greyscaling bad for GANs!

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)

    max_data = len(train_dataloader)
    epochs = np.array([i for i in range(start_epoch, total_epochs)], dtype=int)

    if save_type == "ten_intermediate":
        save_epochs = np.linspace(start_epoch + total_epochs/10, total_epochs-1, 10).astype(int)

    # initialise output loss arrays
    overall_mean_disc_loss = []
    overall_mean_gen_loss = []
    overall_mean_val_disc_loss = ([] if val_split > 0 else None)
    for epoch in np.nditer(epochs):
        # loop through epochs, and initialise outputs per epoch
        epoch_disc_loss = []
        epoch_gen_loss = []
        epoch_val_disc_loss = []
        for idx, data in enumerate(train_dataloader, 0):
            ## 1. update weights of discriminator in respect of the gradient
            net_discriminator.zero_grad()

            if cgan:
                img_labels = Variable(data[1]).to(device)

            ## train discriminiator on real images
            real_img = Variable(data[0]).to(device)
            real_img = (real_img * 2) - 1 # normalise to [-1, 1]
            real_img_with_noise = real_img + torch.randn_like(real_img) * disc_noise_std  # Add noise to a copy of the image tensor
            batch_size = real_img_with_noise.size()[0]
            if label_smoothing > 0: # prevents discriminator from becoming overconfident
                ones = ((1-label_smoothing) * Variable(torch.ones(batch_size))).to(device)
            else:
                ones = Variable(torch.ones(batch_size)).to(device) # label: 1 = real

            if cgan:
                output = net_discriminator.forward(images = real_img_with_noise, labels = img_labels)
            else:
                output = net_discriminator.forward(real_img_with_noise)

            real_img_error = criterion(output, ones)

            ## train discriminiator on fake images
            if cgan:
                noise_tensor = Variable(torch.randn(batch_size, nn_input_size)).to(device)
                fake_img = net_generator.forward(noise = noise_tensor, labels = img_labels)
            else:
                noise_tensor = Variable(torch.randn(batch_size, nn_input_size, 1, 1)).to(device)
                fake_img = net_generator.forward(noise_tensor)
            
            fake_img_with_noise = fake_img + torch.randn_like(fake_img) * disc_noise_std  # Add noise to a copy of the image tensor
            if label_smoothing > 0: # prevents discriminator from becoming overconfident
                zeros = ((label_smoothing) * Variable(torch.ones(batch_size))).to(device)
            else:    
                zeros = Variable(torch.zeros(batch_size)).to(device) # label: 0 = fake
            # detach the gradient from generator (saves computation)
            if cgan:
                output = net_discriminator.forward(images = fake_img_with_noise.detach(), labels = img_labels.detach())
            else:
                output = net_discriminator.forward(fake_img_with_noise.detach())
            fake_img_error = criterion(output, zeros)

            ## backpropagate total error
            descriminator_error = real_img_error + fake_img_error
            descriminator_error.backward()
            optimizer_discriminator.step()

            ## 2. update weights of generator
            net_generator.zero_grad()
            # now we keep the gradient so we can update the weights of the generator
            if cgan:
                output = net_discriminator.forward(images = fake_img, labels = img_labels)
            else:
                output = net_discriminator.forward(fake_img)
            generator_error = criterion(output, ones)
            generator_error.backward()
            optimizer_generator.step()
            
            ## show loss
            this_disc_loss = descriminator_error.data
            this_gen_loss = generator_error.data
            epoch_disc_loss += [this_disc_loss]
            epoch_gen_loss += [this_gen_loss]
            if verbose >= 2:
                print(f'        Progress :: Epoch: {int(epoch+1)}/{total_epochs}, Step: {idx+1}/{max_data}')
                print(f'        Descriminator Loss: {this_disc_loss:.4f} | Generator Loss: {this_gen_loss:.4f}')

            # Save the images if save_type is not "none", but if save_type == "ten_intermediate", then only save if one of the preset 10 epochs    
            if (save_type == "ten_intermediate" and epoch in save_epochs) or save_type in ["final", "all"]:
                if idx == len(train_dataloader) - 1: # Only save images for the final batch of the epoch
                    save_img_verbose = (verbose >= 2)
                    save_images(data[0], 'real_samples', 0, path_prefix = save_path, verbose = save_img_verbose)
                    save_images(fake_img.data, 'fake_samples', int(epoch+1), path_prefix = save_path, verbose = save_img_verbose)
        
        # If using holdout validation, calculate the validation discriminator error
    # Note that the generator loss is only calculated against the fake images, so only validating the discriminator makes sense 
        if val_split > 0:
            with torch.no_grad():
                net_discriminator.eval()
                for val_data in val_dataloader:
                    ## validate discriminiator on real images
                    val_real_img = Variable(val_data[0]).to(device)
                    val_real_img = (val_real_img * 2) - 1 # normalise to [-1, 1]
                    batch_size = val_real_img.size()[0]
                    if label_smoothing > 0: # prevents discriminator from becoming overconfident
                        ones = ((1-label_smoothing) * Variable(torch.ones(batch_size))).to(device)
                    else:    
                        ones = Variable(torch.ones(batch_size)).to(device) # label: 1 = real
                    if cgan:
                        val_labels = Variable(val_data[1]).to(device)
                        val_output = net_discriminator.forward(images = val_real_img.detach(), labels = val_labels)
                    else:
                        val_output = net_discriminator.forward(val_real_img.detach())
                    val_real_img_error = criterion(val_output, ones)

                    val_descriminator_error = val_real_img_error # + fake_img_error
                    this_val_disc_loss = val_descriminator_error.data
                    epoch_val_disc_loss += [this_val_disc_loss]

                    if verbose >= 2:
                        print(f'        Descriminator Val Loss: {this_val_disc_loss:.4f}')
                net_discriminator.train()

        # show higher level loss, by taking mean across batches
        mean_disc_loss = torch.mean(torch.Tensor(epoch_disc_loss))
        mean_gen_loss = torch.mean(torch.Tensor(epoch_gen_loss))
        overall_mean_disc_loss += [mean_disc_loss]
        overall_mean_gen_loss += [mean_gen_loss]

        if val_split > 0:
            mean_val_disc_loss = torch.mean(torch.Tensor(epoch_val_disc_loss))
            overall_mean_val_disc_loss += [mean_val_disc_loss]
        
        if verbose >= 1:
            print(f'    Progress :: Epoch: {int(epoch+1)}/{total_epochs}')
            print(f'    Mean Descriminator Loss: {mean_disc_loss:.4f} | Mean Generator Loss: {mean_gen_loss:.4f}')
            if val_split > 0:
                print(f'        Descriminator Val Loss: {mean_val_disc_loss:.4f}')
        
        # Set up state dicts for the models to return/save
        disc_state_dict = {'epoch': epoch + 1,
            'state_dict': net_discriminator.state_dict(),
            'optimizer': optimizer_discriminator.state_dict()}
        gen_state_dict = {'epoch': epoch + 1,
            'state_dict': net_generator.state_dict(),
            'optimizer': optimizer_generator.state_dict()}
        
        if (save_type == "all" or (save_type == "ten_intermediate" and epoch in save_epochs)):
            save_model(disc_state_dict, 'discriminator', int(epoch + 1), path_prefix = save_path)
            save_model(gen_state_dict, 'generator', int(epoch + 1), path_prefix = save_path)

    if save_type == "final":
        save_model(disc_state_dict, 'discriminator', "final", path_prefix = save_path)
        save_model(gen_state_dict, 'generator', "final", path_prefix = save_path)

    return disc_state_dict, gen_state_dict, overall_mean_disc_loss, overall_mean_gen_loss, overall_mean_val_disc_loss

def create_gan_imgs(gen_model, n_syn, model_save_type = "state_dict", parent_path = "gan_training", nn_input_size = 100, 
                                nn_n_feat = 64, nn_n_layers_gen = 5, model_type = "v2", seed = None, n_classes = None):
    # model_type either path to the models, or state_dict of the neural net (and optimiser if needed)
    # Assuming multiple real image classes, with 1 sub-GAN per class
  
    cgan = False # Haven't coded cGANs yet
    label_list = gen_model.keys()
    n_classes = len(label_list)
    
    n_syn_class = int(np.ceil(n_syn/n_classes))
    synthetic_images_dict = {}
    for label in label_list:
        # First, load in the generator models
        # initialise
        if model_type == "v2":
            net_generator = Generator_v2(input_size = nn_input_size, n_feat = nn_n_feat, n_layers = nn_n_layers_gen).apply(weights_init)
        else:
            raise ValueError(f"Unrecognised model_type input {model_type}")
        
        if model_type == "v2":
            model = gen_model[label]
            optimizer_generator = optim.Adam(net_generator.parameters(), lr=2e-4, betas=(.5, .999)) #### not using if not training further, but needed for the load_model function to work
            if model_save_type == "path":
                # model needs to be the name of the generator, e.g. dcgan_generator_100.pth
                path = f"{parent_path}/label_{label}/model/{model}"
                net_generator, optimizer_generator, _ = load_model(path, net_generator, optimizer_generator)
            elif model_save_type == "state_dict":
                net_generator.load_state_dict(model['state_dict'])
                # optimizer_generator.load_state_dict(model['optimizer']) ##### don't need
            elif model_save_type == "scripted":
                net_generator.load_state_dict(model)
                # net_generator.load_state_dict(torch.load("generator_state_dict.pth", map_location="mps"))
            else:
                raise ValueError(f"Unrecognised model_save_type {model_save_type}")
            
        if seed is not None:
            torch.manual_seed(seed)
        if cgan:
            noise_tensor = Variable(torch.randn(n_syn_class, nn_input_size))
            img_labels = torch.tensor([label] * n_syn_class)
            label_syn_imgs = net_generator.forward(noise = noise_tensor, labels = img_labels)
        else:
            noise_tensor = Variable(torch.randn(n_syn_class, nn_input_size, 1, 1))
            label_syn_imgs = net_generator.forward(noise_tensor)

        label_syn_imgs = (label_syn_imgs + 1) / 2 ### rescale created images back to [0, 1] - standard range
        synthetic_images_dict[label] = label_syn_imgs.detach()

    return synthetic_images_dict

def test_gan(gen_model_version, reps = 1, model_type = "v2", gen_model_parent_path = "gan_training", n_classes = 10, 
             methods = ["fid"], device_name = "cpu", nn_input_size = 100, nn_n_feat = 64, nn_n_layers_gen = 5,
             class_n_real = 1024, class_drop_conv = 0.5, class_aug_ratio = 4, verbose = False):
    assert (isinstance(reps, int) and reps > 0), "reps must be a positive integer"

    # Load the CIFAR-10 data to memory
    full_cifar_dataset = CIFAR10
    # Define data transformations
    transform = transforms.Compose([transforms.ToTensor(),]) # Convert PIL Image to PyTorch Tensor

    train_dataset = full_cifar_dataset(
        root=f'./{"data/cifar-10-orig"}',  # Change the root directory as needed
        train=True,      # Set to True for the training set
        transform=transform,
        target_transform=torch.tensor,  # Convert labels to tensors
        download=True)

    test_dataset = full_cifar_dataset(
        root=f'./{"data/cifar-10-orig"}',  # Change the root directory as needed
        train=False,     # Set to False for the test set
        transform=transform,    
        target_transform=torch.tensor,  # Convert labels to tensors
        download=True)

    device = torch.device(device_name)
    # First of all, load in our data and models
    classes = np.arange(n_classes) # for now assume cifar10

    gen_model = {}
    for label in classes:
        path_to_model = f"{gen_model_parent_path}/label_{label}/model/dcgan_generator_{gen_model_version}.pth"
        gen_model[label] = torch.load(path_to_model, map_location=device)#, strict=False)

    all_reps_scores_dict = {}
    for rep in range(reps):
        if reps > 1:
            print(f"Progress :: Calculating scores for rep {rep+1} of {reps}.")
        rep_scores_dict = {}
        for test_method in methods:
            if verbose:
                print(f"Progress :: Calculating score for {test_method} method" + (f" for rep {rep+1} of {reps}" if reps > 1 else ""))
            if test_method == "class_rel_acc":
                class_train_dataset_samp, _, _ = train_sampler(train_dataset, class_n_real, seed = rep)
                # Get synthetic data ready
                class_n_syn = class_n_real * class_aug_ratio
                # class_synthetic_images_dict = test_synthetic_images_dict
                class_synthetic_images_dict = create_gan_imgs(gen_model, n_syn = class_n_syn, model_save_type = "state_dict", model_type=model_type,
                                                            nn_input_size=nn_input_size, nn_n_feat=nn_n_feat, nn_n_layers_gen=nn_n_layers_gen, n_classes = n_classes)

                classifer_CNN_params = CNN_params_setup(device)
                dropout_fc = classifer_CNN_params["dropout_fc"]
                dropout_conv = class_drop_conv
                classifier_model = Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)
                # Calculate baseline accuracy
                _, _, baseline_test_acc = nn_trainer(classifier_model, class_train_dataset_samp, test_dataset, device_str = device_name, 
                                                                verbose = False, CNN_params_dict=classifer_CNN_params, augmentation=False)
                
                class_syn_dataset = syn_dict_to_dataset(class_synthetic_images_dict)
                class_aug_dataset = ConcatDataset([class_train_dataset_samp, class_syn_dataset])

                # Calculate augmented accuracy
                classifier_model = Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)
                _, _, aug_test_acc = nn_trainer(classifier_model, class_aug_dataset, test_dataset, device_str = device_name, 
                                                                verbose = False, CNN_params_dict=classifer_CNN_params, augmentation=False)
                relative_accuracy = aug_test_acc/baseline_test_acc
                rep_scores_dict[test_method] = relative_accuracy

            if test_method in ["fid", "fid_inf"]:
                real_imgs = np.stack([item[0] for item in test_dataset])
                #real_imgs_flattened = real_imgs.reshape(n_real, -1)
                # Generate synthetic
                n_syn = len(test_dataset)
                syn_images_dict = create_gan_imgs(gen_model, n_syn = n_syn, model_save_type = "state_dict", model_type=model_type,
                                                            nn_input_size=nn_input_size, nn_n_feat=nn_n_feat, nn_n_layers_gen=nn_n_layers_gen, n_classes = n_classes)
                syn_dataset = syn_dict_to_dataset(syn_images_dict)
                syn_imgs = np.stack([item[0] for item in syn_dataset])
                #syn_imgs_flattened = syn_imgs.reshape(n_syn, -1)

                real_tensor, syn_tensor = torch.tensor(real_imgs), torch.tensor(syn_imgs)

                if test_method == "fid":
                    fid_score = calculate_fid_array(real_tensor, syn_tensor, device_name=device_name)
                    rep_scores_dict[test_method] = fid_score
                elif test_method == "fid_inf":
                    fid_inf_score = calculate_FID_infinity_array(real_tensor, syn_tensor, min_fake=999, num_points=15)
                    rep_scores_dict[test_method] = fid_inf_score

        all_reps_scores_dict[rep] = rep_scores_dict

    if reps == 1:
        return rep_scores_dict
    else:
        # Initialize the summary dictionary
        summary_dict = {'mean': {}, 'std': {}}

        # Calculate the mean and standard deviation for each metric
        for sub_dict in all_reps_scores_dict.values():
            for key, value in sub_dict.items():
                summary_dict['mean'].setdefault(key, []).append(value)

        # Take the mean and standard deviation of each metric
        for key, value in summary_dict['mean'].items():
            summary_dict['mean'][key] = np.mean(value)
            summary_dict['std'][key] = np.std(value)

        return summary_dict
    
def plot_gan_output(gen_model, n_syn = 2, model_type = "v2", label_dict = None, model_save_type = "state_dict",
        nn_input_size = 100, nn_n_layers_gen = 5, n_classes = None, seed = None, wrap_single = False, 
        man_title = None, plot_titles = True, save_name = None):

    label_list = gen_model.keys()
    n_classes = len(label_list)

    total_n_syn = n_syn * n_classes
    test_synthetic_images_dict = create_gan_imgs(gen_model, n_syn = total_n_syn, model_save_type = model_save_type, model_type = model_type, 
                                                 nn_input_size = nn_input_size, nn_n_layers_gen = nn_n_layers_gen, n_classes = n_classes, seed = seed)

    if n_syn == 1 and wrap_single:
        ncol = int(np.floor(len(label_list) / 2))
        nrow = total_n_syn // ncol
        fig, axes = plt.subplots(nrows = nrow, ncols = ncol, figsize = (4*ncol, 1+3*nrow), sharey = "row")
        for j, (label, imgs) in enumerate(test_synthetic_images_dict.items()):
            img = imgs[0]
            i = int(j >= 5)
            j = j % 5
            ax = axes[i, j]
            ax.imshow(np.array(img).transpose(1, 2, 0))
            ax.axis('off')  # Optional: Turn off axis
            if label_dict is not None:
                ax.set_title(f"{label_dict[label].decode('utf-8')}", fontsize=34)
            else: 
                ax.set_title(f"{label}", fontsize=34)
            # if label_dict is not None:
            #     ax.set_title(f"Image: {label_dict[label].decode('utf-8')}", fontsize=34)
            # else: 
            #     ax.set_title(f"Image: {label}", fontsize=34)

            plt.subplots_adjust(hspace = 0.2)
        
    else:
        ncol = len(label_list)
        nrow = total_n_syn // ncol
        fig, axes = plt.subplots(nrows = nrow, ncols = ncol, figsize = (4*ncol, 1+3*nrow), sharey = "row")
        for j, (label, imgs) in enumerate(test_synthetic_images_dict.items()):
            for i, img in enumerate(imgs):
                if nrow > 1:
                    ax = axes[i, j]
                else:
                    ax = axes[j]
                ax.imshow(np.array(img).transpose(1, 2, 0))
                ax.axis('off')  # Optional: Turn off axis
                if i == 0:
                    if label_dict is not None:
                        ax.set_title(f"{label_dict[label].decode('utf-8')}", fontsize=34)
                    else: 
                        ax.set_title(f"{label}", fontsize=34)
                    # if label_dict is not None:
                    #     ax.set_title(f"Image: {label_dict[label].decode('utf-8')}", fontsize=34)
                    # else: 
                    #     ax.set_title(f"Image: {label}", fontsize=34)
    title_text = man_title if man_title is not None else "Sample images from GAN"
    y_height = 1.03 if wrap_single else 1.1
    if plot_titles:
        plt.suptitle(title_text, fontsize=36, y=y_height)
        
    if save_name is not None:
        plt.savefig(save_name, format="pdf", bbox_inches="tight")

    plt.show()
    
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular 
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # Trying fix from here: https://github.com/lucidrains/denoising-diffusion-pytorch/issues/213
    covmean = linalg.fractional_matrix_power(sigma1.dot(sigma2), 0.5)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_activations_from_arr(images, model, batch_size, dims, device):
    model.eval()
    n_samples = images.shape[0]
    pred_arr = np.empty((n_samples, dims))
    start_idx = 0
    for i in range(0, n_samples, batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr

def calculate_activation_statistics_from_arrays(images, model, batch_size, dims, device):
    """ 
        Alteration to the file `calculate_activation_statistics` from fid_score, to calculate the activation statistics from arrays of images
    """
    pred_arr = calculate_activations_from_arr(images, model, batch_size, dims, device)

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

def calculate_fid_array(real_arr, fake_arr, device_name="cpu", verbose = False):
    # Preset FID inception model parameters
    device = torch.device(device_name)
    incep_dims = 2048; incep_batch_size = 50
    incep_mod = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[incep_dims]]).to(device)
    if verbose:
        print("Progress :: For validation by FID, calculating baseline inception statistics on the real images.")
    mu_true, sigma_true = calculate_activation_statistics_from_arrays(real_arr, incep_mod, incep_batch_size, incep_dims, device)
    if verbose:
        print("Progress :: Calculating FID scores on the synthetic data")
    mu, sigma = calculate_activation_statistics_from_arrays(fake_arr, incep_mod, incep_batch_size, incep_dims, device)
    fid = calculate_frechet_distance(mu_true, sigma_true, mu, sigma)
    return fid


def get_activations(dataloader, model):
    """
    Get inception activations from dataset
    """
    pool = []
    logits = []

    for images in dataloader:
        # images = images.cuda()
        images = images.to(torch.device("mps"))
        with torch.no_grad():
            pool_val, logits_val = model(images)
            pool += [pool_val]
            logits += [F.softmax(logits_val, 1)]

    return torch.cat(pool, 0).cpu().numpy(), torch.cat(logits, 0).cpu().numpy()

# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception,self).__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)
    def forward(self, x):
        x = (x - self.mean) / self.std
        # Upsample if necessary
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        # 299 x 299 x 3
        x = self.net.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.net.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.net.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.net.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.net.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.net.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.net.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.net.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.net.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.net.Mixed_6e(x)
        # 17 x 17 x 768
        # 17 x 17 x 768
        x = self.net.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.net.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.net.Mixed_7c(x)
        # 8 x 8 x 2048
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        # 1 x 1 x 2048
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        # 1000 (num_classes)
        return pool, logits

def load_inception_net(parallel=False):
    inception_model = inception_v3(pretrained=True, transform_input=False)
    #inception_model = WrapInception(inception_model.eval()).cuda()
    inception_model = WrapInception(inception_model.eval()).to(torch.device("mps"))
    if parallel:
        inception_model = nn.DataParallel(inception_model)
    return inception_model

def FID_inf_from_acts(real_m, real_s, fake_act, min_fake, num_points):
    # Utility function to calculate the FID_inf from real mu & sigma, and synthetic activations.
    # Separated out to use in multiple functions
    num_fake = len(fake_act)
    assert num_fake > min_fake, \
        'number of fake data must be greater than the minimum point for extrapolation'

    # Choose the number of images to evaluate FID_N at regular intervals over N
    fid_batches = np.linspace(min_fake, num_fake, num_points).astype('int32')
    fids = []
    # Evaluate FID_N
    for fid_batch_size in fid_batches:
        # sample with replacement
        np.random.shuffle(fake_act)
        fid_activations = fake_act[:fid_batch_size]
        m, s = np.mean(fid_activations, axis=0), np.cov(fid_activations, rowvar=False)
        FID = calculate_frechet_distance(m, s, real_m, real_s)
        fids.append(FID)
    fids = np.array(fids).reshape(-1, 1)
    
    # Fit linear regression
    reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
    fid_infinity = reg.predict(np.array([[0]]))[0,0]
    return fid_infinity


def calculate_FID_infinity_array(real_arr, fake_arr, min_fake=5000, num_points=15, device_name="cpu"):
    """
    Rework of this function to use the pytorch-fid implementation.
    Calculates effectively unbiased FID_inf using extrapolation given 
    arrays (currently working with torch tensors) of real & synthetic data
    Args:
        real_array: (arr) 
            An array of real data, n images in 3 colour channels, shape:
            n by 3 by x by y 
        fake_array: (arr)
            An array of fake data, m images in 3 colour channels, shape:
            m by 3 by x by y 
        min_fake: (int)
            Minimum number of images to evaluate FID on.
            Default: 5000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    """
    # Setup the inception model
    #device_name = "mps"
    incep_batch_size = 50
    incep_dims = 2048
    device = torch.device(device_name)
    incep_mod = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[incep_dims]]).to(device)

    real_m, real_s = calculate_activation_statistics_from_arrays(real_arr, incep_mod, incep_batch_size, incep_dims, device)
    fake_act = calculate_activations_from_arr(fake_arr, incep_mod, incep_batch_size, incep_dims, device)
    # This function takes the real mu & sigma, and initial synthetic activations, and calculates FID infinity from those
    fid_infinity = FID_inf_from_acts(real_m, real_s, fake_act, min_fake, num_points)
    return fid_infinity

def calculate_fid_infs(synthetic_datasets, images_true, device, min_fake=5000, num_points=15, verbose = False):
    """Calculate the FID_inf of sets of synthetic data against a set of real data
    Args:
        synthetic_datasets (dict): The synthetic images to assess FID for. Dict of datasets keyed by synthetic image type, each element is n x 3 x 32 x 32 array of n images.
        images_true (tensor): Real images to compare FID against
        device (torch device): Device to run computing on
        min_fake: (int)
            Minimum number of images to evaluate FID on.
            Default: 5000
        num_points: (int)
            Number of FID_N we evaluate to fit a line.
            Default: 15
    Returns:
        dict: Dictionary of FID_inf values for each synthetic set
    """
    # Preset FID inception model parameters
    incep_dims = 2048; incep_batch_size = 50
    incep_mod = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[incep_dims]]).to(device)
    if verbose:
        print("Progress :: For validation by FID_inf, calculating baseline inception statistics on the real images.")
    mu_true, sigma_true = calculate_activation_statistics_from_arrays(images_true, incep_mod, incep_batch_size, incep_dims, device)
    if verbose:
        print("Progress :: Calculating FID_inf scores on the synthetic data")
    fid_inf_dict = {}
    for syn_set, data in synthetic_datasets.items(): # loop through the input synthetic datasets
        # Calculate the activation statistics for each synthetic set, then the frechet distance of that against the baseline real data, and save to fid_inf_dict, keyed by set_name
        fake_arr = torch.from_numpy(np.array(data).reshape(-1, 3, 32, 32)).float()
        fake_act = calculate_activations_from_arr(fake_arr, incep_mod, incep_batch_size, incep_dims, device)
        fid_inf_dict[syn_set] = FID_inf_from_acts(mu_true, sigma_true, fake_act, min_fake, num_points)
    return fid_inf_dict