import os
import json
import numpy as np
import torchvision
import torchvision.transforms as transforms
from docopt import docopt
from time import time

import utils

def import_data_to_mem(dataset_name = "cifar10", test_data_needed = False):
    """ Import the given data from the torchvision or fgvc dataset to memory.
    Args:
        dataset_name (string, optional): The name of the dataset to import. Currently accepting "cifar10" or "fgvc". Defaults to "cifar10".
        test_data_needed (bool, optional): Whether also need to return a test dataset from CIFAR. Defaults to False.
    Returns:
        train_dataset, test_dataset: Training (and Test if required) datasets from torchvision CIFAR
    """
    # Define data transformations
    transform = transforms.Compose([transforms.ToTensor(),]) # Convert PIL Image to PyTorch Tensor

    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    else:
        raise ValueError(f"dataset name {dataset_name} not recognised")
    
    # Download and load the dataset
    train_dataset = dataset(
        root=f'./{dataset_name}',  # Change the root directory as needed
        train=True,      # Set to True for the training set
        transform=transform,
        download=True)

    if test_data_needed:
        test_dataset = dataset(
            root=f'./{dataset_name}',  # Change the root directory as needed
            train=False,     # Set to False for the test set
            transform=transform,
            download=True)
    else: 
        test_dataset = []
    return train_dataset, test_dataset

def train_gan_per_class(sampled_train_dataset_by_class, epochs, out_path, save_type, train_verb, val_split, batch_size, loader_num_workers, device_name,
        lr_gen, lr_disc, beta_1, beta_2, wd, input_size, n_layers_gen, n_layers_disc, p_drop, label_smoothing, disc_noise_std, augmentation = False, aug_ratio = 2, aug_var = 3):
    #disc_models = {}
    #gen_models = {}

    mean_disc_loss_dict = {}
    mean_gen_loss_dict = {}
    mean_val_disc_loss_dict = {}
    
    for label, dataset_samp in sampled_train_dataset_by_class.items():
        t0 = time() # For timing the code
        if train_verb > 0:
            print(f"Progress :: Training GAN for label {label}")
        save_path = f"{out_path}/label_{label}"
        _, _, label_mean_disc_loss, label_mean_gen_loss, label_mean_val_disc_loss = utils.train_gan(
                dataset_samp, total_epochs = epochs, save_path = save_path, save_type = save_type, verbose=train_verb, val_split = val_split,
                batch_size=batch_size, loader_num_workers=loader_num_workers, device_name=device_name, lr_gen=lr_gen, lr_disc=lr_disc, 
                opt_beta_1=beta_1, opt_beta_2=beta_2, weight_decay = wd, nn_input_size = input_size, nn_n_layers_gen = n_layers_gen, nn_n_layers_disc = n_layers_disc, 
                nn_p_drop = p_drop, label_smoothing=label_smoothing, disc_noise_std=disc_noise_std, augmentation = augmentation, aug_ratio = aug_ratio, aug_var = aug_var)
#        disc_models[int(label)] = disc_mod_state_dict
#        gen_models[int(label)] = gen_mod_state_dict
        mean_disc_loss_dict[str(label)] = [tensor.item() for tensor in label_mean_disc_loss]
        mean_gen_loss_dict[str(label)] = [tensor.item() for tensor in label_mean_gen_loss]
        if val_split > 0:
            mean_val_disc_loss_dict[str(label)] = [tensor.item() for tensor in label_mean_val_disc_loss]
        t1 = time()
        if train_verb > 0:
            print(f"    That took {(t1-t0)/60:.1f} minutes.")
    return mean_disc_loss_dict, mean_gen_loss_dict, mean_val_disc_loss_dict

def save_logs_to_jsons(out_path, mean_disc_loss, mean_gen_loss, mean_val_disc_loss, dict_out_type = True):
    print(f"Progress :: Saving the outputs (models and training logs) in folder {out_path}")
    #json.dump(disc_models, open(f"{out_path}/disc_models.json", 'w'))
    #json.dump(gen_models, open(f"{out_path}/gen_models.json", 'w'))
    if dict_out_type:
        json.dump(mean_disc_loss, open(f"{out_path}/mean_disc_loss_dict.json", 'w'))
        json.dump(mean_gen_loss, open(f"{out_path}/mean_gen_loss_dict.json", 'w'))
        json.dump(mean_val_disc_loss, open(f"{out_path}/mean_val_disc_loss_dict.json", 'w')) # hopefully this works if val_split == 0
    else:
        json.dump({"all_labels" : [tensor.item() for tensor in mean_disc_loss]}, open(f"{out_path}/mean_disc_loss_dict.json", 'w'))
        json.dump({"all_labels" : [tensor.item() for tensor in mean_gen_loss]}, open(f"{out_path}/mean_gen_loss_dict.json", 'w'))
        json.dump({"all_labels" : [tensor.item() for tensor in mean_val_disc_loss]}, open(f"{out_path}/mean_val_disc_loss_dict.json", 'w'))

def test_model_from_path(test_methods, save_type, epochs, out_path, device_name, input_size, n_layers_gen, test_reps, mmd_sigma = 12, class_n_real = 1024, class_drop_conv = 0.5, dist_n_real = 4000, gan_type = "v2"):
    t0 = time() # For timing the code
    print(f"Testing the model using the following methods: {test_methods}")
    model_version = "final" if save_type == "final" else epochs
    test_scores_dict = utils.test_gan(gen_model_version = model_version, gen_model_parent_path = out_path, methods = test_methods, device_name=device_name,
                                            nn_input_size = input_size, nn_n_layers_gen = n_layers_gen, reps = test_reps, verbose = (test_reps == 1), 
                                            mmd_sigma = mmd_sigma, class_n_real = class_n_real, class_drop_conv = class_drop_conv, dist_n_real = dist_n_real, model_type=gan_type)
    json.dump(test_scores_dict, open(f"{out_path}/test_scores.json", 'w'))
    t1 = time()
    print(f"    That took {(t1-t0)/60:.1f} minutes.")

def train_save_test_model(params, sampled_train_dataset_by_class, test_methods, test_reps = 5, augmentation = False, aug_ratio = 2, aug_var = 3):
    mean_disc_loss_dict, mean_gen_loss_dict, mean_val_disc_loss_dict = train_gan_per_class(sampled_train_dataset_by_class, 
                                            **params, augmentation = augmentation, aug_ratio = aug_ratio, aug_var = aug_var)
    
    save_logs_to_jsons(params['out_path'], mean_disc_loss_dict, mean_gen_loss_dict, mean_val_disc_loss_dict)
    test_model_from_path(test_methods, params['save_type'], params['epochs'], params['out_path'], 
                        params['device_name'], params['input_size'], params['n_layers_gen'], 
                        test_reps=test_reps, class_n_real = 1000, dist_n_real = 1000)

if __name__ == "__main__":    
    # Predefine docopt options for command line interface
    __doc__ = """Train a GAN model

    Usage:
    train_cifar_gan.py [options]

    Options:
    -h --help                    Show this screen.
    --dataset_name=<set str>     The name of the dataset to import. Currently accepting "cifar10" (plan to generalise to include "fgvc"). [default: cifar10]
    --out_path=<output fldr>     Location to save the output images to [default: gan_output]
    --save_type=<type str>       How the gan outputs should be saved. Options: "final", "all", "ten_intermediate". [default: ten_intermediate]
    --cond_gan=<gan type bool>   Whether training a cGAN (conditional), or standard GAN. If training a standard GAN, split the data by class, and calculate n_class mini-GANs. Boolean. [default: False]
    --samples=<n_real>           Number of real training samples. Must be <= 50,000. [default: 1024]
    --split_before_samp=<bool>   Whether or not to split the data by class (if not cGAN) before sampling. Ensures even classes, but doesn't give same sample as other SD experiments. [default: True]
    --batch_size=<n_batch>       Size of batches for nn training. [default: 64]
    --epochs=<n_epochs>          Number of epochs to use for nn training. [default: 50]
    --val_split=<ratio>          Ratio of data to holdout for discriminator validation. [default: 0]
    --num_workers=<n_workers>    How many workers to use in the dataloader. [default: 2]
    --lr_gen=<gan_param>         Learning rate for the generator. [default: 2e-4]
    --lr_disc=<gan_param>        Learning rate for the discriminator. [default: 2e-4]
    --beta_1=<gan_param>         Optimiser beta_1 parameter for the GAN. [default: 0.5]
    --beta_2=<gan_param>         Optimiser beta_2 parameter for the GAN. [default: 0.999]
    --wd=<gan_param>             Weight decay parameter for the GAN. [default: 0]
    --input_size=<gan_param>     Number of nodes in the generator input layer parameter for the GAN. [default: 100]
    --n_layers_gen=<gan_param>   Number of layers in the generator parameter for the GAN. [default: 5]
    --n_layers_disc=<gan_param>  Number of layers in the discriminator parameter for the GAN. [default: 5]
    --p_drop=<gan_param>         Dropout parameter for the GAN. [default: 0]
    --label_smth=<gan_param>     Parameter to smooth the discriminator labels to reduce overconfidence. Probably anything above 0.2 is risky. [default: 0]
    --disc_noise=<gan_param>     Standard deviation of noise to add to images to feed into discriminator to minimise vanishing gradient. [default: 0]
    --training_verbosity=<n_v>   How much information to print from the GAN training. Even on 0, will still give a little high-level info. [default: 0]
    --device_name=<dev str>      The name of the device to train the model on. Accepts cpu, mps or cuda. [default: cpu]
    --test_methods=<list>        List of methods to use to test the GAN on. Comma separated list of class_rel_acc,fid,fid_inf,final_avg_loss, or None to not test. [default: None]
    --test_reps=<reps_int>       The number of experiment repeats to use for testing the gan output with above methods. [default: 1]
    --class_drop_conv=<dropout>  The dropout_conv parameter to use in testing the classifier. [default: 0.5]
    --augmentation=<aug_bool>    Whether non-synthetic (transformational) image augmentations are to be used. Boolean. [default: False]
    --aug_ratio=<data_amt>       If augmentation, the proportion of augmented images to create. Float, must be >= 1. [default: 2]
    --aug_var=<aug_intns>        If augmentation, the transform variability parameter to use, typically a value between 0 and 5 (0 for no augmentations). Float. [default: 3]
    """

    args = docopt(__doc__)
    
    dataset_name = args['--dataset_name']
    out_path = args['--out_path']
    save_type = args['--save_type']
    cond_gan = True if args['--cond_gan'] == "True" else False
    N = int(args['--samples'])
    split_before_samp = True if args['--split_before_samp'] == "True" else False
    batch_size = int(args['--batch_size'])
    epochs = int(args['--epochs'])
    val_split = float(args['--val_split'])
    lr_gen = float(args['--lr_gen'])
    lr_disc = float(args['--lr_disc'])
    beta_1 = float(args['--beta_1'])
    beta_2 = float(args['--beta_2'])
    wd = float(args['--wd'])
    input_size = int(args['--input_size'])
    n_layers_gen = int(args['--n_layers_gen'])
    n_layers_disc = int(args['--n_layers_disc'])
    p_drop = float(args['--p_drop'])
    label_smoothing = float(args['--label_smth'])
    disc_noise_std = float(args['--disc_noise'])
    loader_num_workers = int(args['--num_workers'])
    train_verb = float(args['--training_verbosity'])
    device_name = args['--device_name']
    test_methods = None if args['--dataset_name'] == "None" else [m for m in args['--test_methods'].split(',')]
    test_reps = int(args['--test_reps'])
    class_drop_conv = float(args['--class_drop_conv'])
    augmentation = True if args['--augmentation'] == "True" else False
    aug_var = float(args['--aug_var']) if args['--aug_var'] != "None" else None
    aug_ratio = float(args['--aug_ratio'])

    t00 = time() # For timing the code

    print(f"Using {device_name} device for computations")

    n_classes = 10 #constant for cifar10

    print(f"Progress :: Setting up data")
    train_dataset, _ = import_data_to_mem(dataset_name)

    if augmentation:
        out_path = f"{out_path}_aug_ratio_{aug_ratio}"

    # Otherwise, split and sample it by class
    gan_type = "v2"
    n_class_samp = N // n_classes
    cifar10_classes = np.arange(n_classes)
    if split_before_samp:
        train_dataset_by_class = utils.split_dataset_by_class(train_dataset, cifar10_classes)
        sampled_train_dataset_by_class = utils.sample_split_datasets(train_dataset_by_class, n_class_samp)
    else:
        sampled_train_dataset, _, _ = utils.train_sampler(train_dataset, N, seed = 42)
        sampled_train_dataset_by_class = utils.split_dataset_by_class(sampled_train_dataset, cifar10_classes)

    mean_disc_loss, mean_gen_loss, mean_val_disc_loss = train_gan_per_class(sampled_train_dataset_by_class, epochs, out_path, save_type, train_verb, val_split, batch_size, loader_num_workers, device_name,
                            lr_gen, lr_disc, beta_1, beta_2, wd, input_size, n_layers_gen, n_layers_disc, p_drop, label_smoothing, disc_noise_std, augmentation, aug_ratio, aug_var)
    dict_out_type = True

    save_logs_to_jsons(out_path, mean_disc_loss, mean_gen_loss, mean_val_disc_loss, dict_out_type = dict_out_type)

    if test_methods is not None:
        test_model_from_path(test_methods, save_type, epochs, out_path, device_name, input_size, n_layers_gen, test_reps, class_drop_conv = class_drop_conv, gan_type = gan_type)

    t11 = time()
    print(f"Total time taken: {(t11-t00)/60:.1f} minutes.")