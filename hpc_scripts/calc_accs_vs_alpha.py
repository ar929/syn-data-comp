import json
import numpy as np
import pandas as pd
import torch
import random

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Subset, ConcatDataset

import syn_paper_utils as utils

######### SETUP
device_name = utils.get_device()
device = torch.device(device_name)
CNN_params_dict = utils.CNN_params_setup(device)
dropout_conv, dropout_fc = CNN_params_dict['dropout_conv'], CNN_params_dict['dropout_fc']

# Load the CIFAR-10 data to memory
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
images_samp = torch.stack([train_samp[i][0] for i in range(N)])  # Stack the image tensors
labels_samp = torch.tensor([train_samp[i][1] for i in range(N)])  # Convert labels to a tensor

# Return as TensorDataset
subset_train_dataset = TensorDataset(images_samp, labels_samp)

avg_baseline_accuracy = 0.42261 # score hard-coded from full run (easier than saving to a separate file)



######### CONV SCORES
conv_scores_filename = 'scores/conv_10run_scores.json'

gamma = 3
alphas = [1, 3, 5, 7, 9]
alphas = [1, 3, 5, 7, 9, 10, 12, 15, 20]

repeats = 25
recalculate_results = True
if recalculate_results:
    conv_scores_dict = {} 
    for repeat in range(repeats):
        print(f"Progress :: Calculating accuracy scores for conventional augs, repeat {repeat + 1} of {repeats}")

        conv_scores_rep = {}
        for alpha in alphas:
            if alpha == 1:
                conv_scores_rep[alpha] = 1
            else:
                CNN_params_dict = utils.CNN_params_setup(device)
                dropout_conv = CNN_params_dict["dropout_conv"]; dropout_fc = CNN_params_dict["dropout_fc"]
                cnn_model = utils.Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = dropout_conv, dropout_fc = dropout_fc).to(device)
                _, _, test_acc = utils.nn_trainer(cnn_model, subset_train_dataset, test_dataset, opt_type = "adam", CNN_params_dict=CNN_params_dict, 
                                                    loss_type = "nll", lr_sched = None, device_str = device_name, verbose = False, 
                                                    augmentation = True, aug_ratio = alpha, aug_var = gamma)
                conv_scores_rep[alpha] = test_acc
        conv_scores_dict[repeat] = conv_scores_rep
    json.dump(conv_scores_dict, open(conv_scores_filename, 'w'))
else:
    conv_scores_dict = json.load(open(conv_scores_filename, 'r'))

######## GAN 2 SCORES
# repeats = 10
n_classes = 10
cifar10_classes = np.arange(n_classes)
model_version = "final"

label_dict = {0: b'airplane', 1: b'automobile', 2: b'bird', 3: b'cat', 4: b'deer', 5: b'dog', 6: b'frog', 7: b'horse', 8: b'ship', 9: b'truck'} # Output from CIFAR10
scripted_models = True # saves space for sharing on git

gan_mod_2_parent_path = "models_save_space/gan_2"
gan_model_2 = {}
for label in cifar10_classes:
    if scripted_models:
        path_to_model = f"{gan_mod_2_parent_path}/label_{label}/generator_scripted.pth"
        gan_model_2[label] = torch.load(path_to_model, map_location=device)
    else:
        path_to_model = f"{gan_mod_2_parent_path}/label_{label}/model/dcgan_generator_{model_version}.pth"
        gan_model_2[label] = torch.load(path_to_model, map_location=device)

gan_mod_4_parent_path = "models_save_space/gan_4"
gan_model_4 = {}
for label in cifar10_classes:
    if scripted_models:
        path_to_model = f"{gan_mod_4_parent_path}/label_{label}/generator_scripted.pth"
        gan_model_4[label] = torch.load(path_to_model, map_location=device)
    else:
        path_to_model = f"{gan_mod_4_parent_path}/label_{label}/model/dcgan_generator_{model_version}.pth"
        gan_model_4[label] = torch.load(path_to_model, map_location=device)         
gan_2_class_dropout_conv = 0.68
gan_4_class_dropout_conv = 0.25
nn_input_size = 150
nn_n_layers_gen = 5

gan_2_score_file_name = "scores/gan_2_scores.json"

if recalculate_results:      
    gan_2_scores_all = {}
    for repeat in range(repeats):
        print(f"Progress :: Calculating accuracy scores for GAN model 2, repeat {repeat + 1} of {repeats}")
        gan_2_scores = {}

        for alpha in alphas:
            if alpha == 1:
                gan_2_scores[alpha] = 1
            else:
                model_type = "v2"
                class_train_datalist_samp, _, _ = utils.train_sampler(train_dataset, N, seed = seed)
                samp_images, samp_labels = zip(*class_train_datalist_samp)
                # Convert them to tensors if they aren't already
                samp_images = torch.stack(samp_images)  # Stack image tensors into a single tensor
                samp_labels = torch.tensor(samp_labels) # Convert labels to a tensor
                # Create the TensorDataset
                class_train_dataset_samp = TensorDataset(samp_images, samp_labels)
                # Get synthetic data ready
                class_n_syn = N * alpha
                # class_synthetic_images_dict = test_synthetic_images_dict
                class_synthetic_images_dict = utils.create_gan_imgs(gan_model_2, n_syn = class_n_syn, model_save_type = "scripted", model_type=model_type,
                                                            nn_input_size=nn_input_size, nn_n_layers_gen=nn_n_layers_gen, n_classes = n_classes)

                class_syn_dataset = utils.syn_dict_to_dataset(class_synthetic_images_dict)
                class_aug_dataset = ConcatDataset([class_train_dataset_samp, class_syn_dataset])

                classifer_CNN_params = utils.CNN_params_setup(device)
                dropout_fc = classifer_CNN_params["dropout_fc"]

                # Calculate augmented accuracy
                classifier_model = utils.Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = gan_2_class_dropout_conv, dropout_fc = dropout_fc).to(device)
                _, _, aug_test_acc = utils.nn_trainer(classifier_model, class_aug_dataset, test_dataset, device_str = device_name, 
                                                                verbose = False, CNN_params_dict=classifer_CNN_params, augmentation=False)
                relative_accuracy = aug_test_acc/avg_baseline_accuracy
                gan_2_scores[alpha] = relative_accuracy
        gan_2_scores_all[repeat] = gan_2_scores
    json.dump(gan_2_scores_all, open(gan_2_score_file_name, 'w'))
else:
    gan_2_scores_all = json.load(open(gan_2_score_file_name, 'r'))

########## GAN 4 SCORES
gan_4_score_file_name = "scores/gan_4_scores.json"

if recalculate_results:      
    gan_4_scores_all = {}
    for repeat in range(repeats):
        print(f"Progress :: Calculating accuracy scores for GAN model 2, repeat {repeat + 1} of {repeats}")
        gan_4_scores = {}

        for alpha in alphas:
            if alpha == 1:
                gan_4_scores[alpha] = 1
            else:
                model_type = "v2"
                class_train_datalist_samp, _, _ = utils.train_sampler(train_dataset, N, seed = seed)
                samp_images, samp_labels = zip(*class_train_datalist_samp)
                # Convert them to tensors if they aren't already
                samp_images = torch.stack(samp_images)  # Stack image tensors into a single tensor
                samp_labels = torch.tensor(samp_labels) # Convert labels to a tensor
                # Create the TensorDataset
                class_train_dataset_samp = TensorDataset(samp_images, samp_labels)
                # Get synthetic data ready
                class_n_syn = N * alpha
                # class_synthetic_images_dict = test_synthetic_images_dict
                class_synthetic_images_dict = utils.create_gan_imgs(gan_model_4, n_syn = class_n_syn, model_save_type = "scripted", model_type=model_type,
                                                            nn_input_size=nn_input_size, nn_n_layers_gen=nn_n_layers_gen, n_classes = n_classes)

                class_syn_dataset = utils.syn_dict_to_dataset(class_synthetic_images_dict)
                class_aug_dataset = ConcatDataset([class_train_dataset_samp, class_syn_dataset])

                classifer_CNN_params = utils.CNN_params_setup(device)
                dropout_fc = classifer_CNN_params["dropout_fc"]

                # Calculate augmented accuracy
                classifier_model = utils.Cifar_CNN(num_channels = 3, classes = 10, dropout_conv = gan_4_class_dropout_conv, dropout_fc = dropout_fc).to(device)
                _, _, aug_test_acc = utils.nn_trainer(classifier_model, class_aug_dataset, test_dataset, device_str = device_name, 
                                                                verbose = False, CNN_params_dict=classifer_CNN_params, augmentation=False)
                relative_accuracy = aug_test_acc/avg_baseline_accuracy
                gan_4_scores[alpha] = relative_accuracy
        gan_4_scores_all[repeat] = gan_4_scores
    json.dump(gan_4_scores_all, open(gan_4_score_file_name, 'w'))
else:
    gan_4_scores_all = json.load(open(gan_4_score_file_name, 'r'))


####### IGPT SCORES
synthetic_datasets, synthetic_label_sets = utils.synthetic_data_import(folder_path = "data/image_gpt_tidied/synthetic_cifar_seed_42", split_by_type = True)
scores_dict = {}
seed = 42
train_samp = utils.data_sampler(train_dataset, N, rand_seed = seed) # function from igpt_generator file
x_train_true = [item[0] for item in train_samp]
y_train_true = [item[1] for item in train_samp]
images_true = torch.stack(x_train_true)


igpt_scores_filename = 'scores/igpt_6run_scores.csv'
if recalculate_results:
    other_seeds_path="data/image_gpt_tidied/synthetic_cifar_43_to_47.zip"

    scores_dict = {}
    baseline_accs_dict = {}

    # Calculate the baseline real-data accuracy, and the classifier accuracy for each synthetic set
    CNN_params, baseline_test_acc, accuracy_dict = utils.classifier_accuracies(synthetic_datasets, synthetic_label_sets, train_samp, test_dataset, 
                                                    syn_augment=True, images_true=images_true, y_train_true=y_train_true, device=device, device_name=device_name,
                                                    augment=False) # this augment is additional conv-aug on top of the igpt augmentation
    
    scores_dict[42] = accuracy_dict 
    baseline_accs_dict[42] = baseline_test_acc


    synthetic_datasets_by_seed, synthetic_label_sets_by_seed = utils.synthetic_data_import(other_seeds_path, split_by_type = True, split_method = "prompt_rep", seed_subset = True, zipped = True)
    print("Progress :: To plot accuracy error bars, computing classifier accuracies for other synthetic datasets.")
    accuracies_dict = {}
    baseline_accuracies_dict = {}
    fids_dict = {}
    for seed, syn_data_dict in synthetic_datasets_by_seed.items():
        print(f"Progress :: Computing accuracies for seed {seed} out of {list(synthetic_datasets_by_seed.keys())}")
        seed_baseline_test_acc, seed_accuracy_dict = utils.seed_subset_accuracies(seed, syn_data_dict, synthetic_label_sets_by_seed, train_dataset, N, test_dataset, True, 
                                                            images_true, y_train_true, device, device_name, augment=False)

        accuracies_dict[seed] = seed_accuracy_dict
        baseline_accuracies_dict[seed] = seed_baseline_test_acc

    # Convert dictionaries to dataframe
    df_scores = pd.DataFrame(scores_dict).T

    # Divide each value by the corresponding baseline score
    rel_score_df = df_scores.div(pd.Series(baseline_accs_dict), axis=0)

    rel_score_df.to_csv(igpt_scores_filename)
else:
    rel_score_df = pd.read_csv(igpt_scores_filename, index_col = 0)