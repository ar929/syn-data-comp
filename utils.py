import warnings
import numpy as np
import random

from sklearn.metrics import accuracy_score

import torch
import torchvision.transforms as transforms
from torch.nn import Module
from torch.utils.data import random_split, DataLoader, TensorDataset, RandomSampler, ConcatDataset

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
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
    
def img_transformation(data, prob = 0.5, variability = 3, random_erase_greyscale = True, seed = None):
    """ Function to apply non-synthetix 
        Data (tensor): either a 3d image or a 4d tensor of images
        Prob (0 <= float <= 1) chance that any given image will be transformed. Defaults to 0.5.
        variability (float): expects a value between 1 and 5, but isn't restricted to this - can take anything from 0 to 8 at least. Defaults to 3. 
        random_erase_greyscale (Bool, optional): Whether to include random erasures into the image transformations. Defaults to True.
        """
    v = variability # for simplicity
    if v < 0 or v > 5:
        warnings.warn(f"Note unexpected variability value {v}; if not between 0 and 5, may have unexpected effects.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Define common transformations
    transform_list = [transforms.RandomResizedCrop(32, (1-(0.1*v), 1+(0.2*v)), antialias=True),
        transforms.Pad(padding=12, fill=0, padding_mode='symmetric'),
        transforms.RandomAffine(degrees=9 * v, shear=3.3 * v),
        transforms.CenterCrop(32),
        transforms.ColorJitter(brightness=0.08 * v, contrast=0.08 * v, saturation=0.1 * v, hue=(0.06 * v if v <= 5 else 0.5))]
    
    # Add conditional transformations
    if v < 1:
        pass  # No extra transformations for v < 1
    else:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.GaussianBlur(kernel_size=int(np.floor(v/4))*2+1))
    
    if random_erase_greyscale:
        transform_list.append(transforms.RandomGrayscale(p=0.04 * v))
        if v >= 1:
            transform_list.append(transforms.RandomErasing(p = (0.15 * v if v <= 6 else 1), scale=(0.01 * v, 0.05 * v))) 
    
    transform = transforms.Compose(transform_list)
    
    if data.ndim == 3:
        if random.random() < prob:
            out_img = transform(torch.tensor(data))
        else:
            out_img = torch.tensor(data)
        return out_img
    elif data.ndim == 4:
        out_imgs = torch.empty_like(data)
        for i in range(len(data)):
            if random.random() < prob:
                out_img = transform(data[i])
            else:
                out_img = torch.tensor(data)
            out_imgs[i] = out_img
        return out_imgs
    else:
        raise ValueError("Input image has unknown number of dimensions.")

def conv_augmenter(train_data, aug_ratio = 2, aug_var = 3, return_combined = True, labelled_data = True, random_erase_greyscale = True):
    """Function to carry out conventional image augmentation. Uses `img_transformation` function.
    Args:
        train_data (Dataset): Dataset of (real) training images.
        aug_ratio (float, optional): The proportion of augmented images (to real images) to create. Must be >= 1. Defaults to 2.
        aug_var (float, optional): The transform variability parameter to use, typically a value between 0 and 5 (0 for no augmentations). Defaults to 3.
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
            new_img = img_transformation(img, prob = 1, variability = aug_var, random_erase_greyscale = random_erase_greyscale)
            aug_imgs += [new_img.squeeze(0)]; aug_labels += [label]
        aug_data = TensorDataset(torch.stack(aug_imgs), torch.tensor(aug_labels))
    else:
        for img in aug_dataloader:
            new_img = img_transformation(img[0], prob = 1, variability = aug_var, random_erase_greyscale = random_erase_greyscale)
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
        score_type (str, optional): The type of test score to return. Currently takes "f1" or "acc". Defaults to "f1".
        opt_type (str, optional): Name of optimiser function to use. Currently only takes "adam". Defaults to "adam".
        loss_type (str, optional): Name of loss function to use. Currently only takes "nll". Defaults to "nll".
        val_split (float, optional): Proportion of data to use for the validation set (and 1 - val_split for the test set) Overwritten if k_fold_k used. Set to 0 for no validation. Defaults to 0.2.
        val_each_step (bool, optional): Whether to calculate statistics on the validation at each step/epoch; if False, only calculate at the end. Defaults to False.
        epochs (int, optional): Number of epochs to train over. Defaults to 50.
        batch_size (int, optional): Size of batches to split the data into for training. Defaults to 64.
        learning_rate (float, optional): Learning rate parameter for the optimiser. Defaults to 1e-3.
        lr_sched (str, optional): If not None, type of learning rate lr_scheduler (currently only takes None or not for a ReduceLROnPlateau). Defaults to None.
        augmentation (bool, optional): Whether non-synthetic (transformational) image augmentations are to be used. Defaults to False.
        aug_ratio (float, optional): If augmentation, the proportion of augmented images to create. Must be = 1. Defaults to 2.
        aug_var (float, optional): If augmentation, the transform variability parameter to use, typically a value between 0 and 5 (0 for no augmentations). Defaults to 3.
        weight_decay (float, optional): Weight decay parameter for the optimiser. Defaults to 0.
        device_str (str, optional): Name of the device to be used for model training. Defaults to "cpu".
        verbose (bool, optional): Whether updates should be given during training process. Defaults to True.
        CNN_params_dict (dict, optional): Can be helpful in other code to input all of these parameters in a single dict. If provided this will overwrite the other parameters. 
                                          Parameters included are epochs, batch_size, learning_rate, val_split, weight_decay and device. Defaults to None.
    Returns:
        model (PyTorch model): Final trained neural network model.
        val_score (float): Final validation score from the training, to use for hyperparameter optimisation. Returns score of choice (score_type) as a value between 0 and 1.
        test_score (float): Test score from the training. Returns score of choice (score_type) as a value between 0 and 1.
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