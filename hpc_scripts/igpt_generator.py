
import os
import utils
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from docopt import docopt
import fgvcdata

def img_resize_const_ratio(img, target_width, target_height):
    # Ensure image in correct format
    img_arr = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))    
    orig_height, orig_width = img_arr.shape[:2]
    # Calculate aspect ratios
    asp_ratio_orig = orig_width/orig_height
    asp_ratio_targ = target_width/target_height

    # Determine new dims for resizing while maintaining aspect ratio
    if asp_ratio_orig > asp_ratio_targ:
        # crop the longer dimension = width
        new_width = int(target_height * asp_ratio_orig)
        resized_img_arr = cv2.resize(img_arr, (new_width, target_height), interpolation=cv2.INTER_CUBIC)#, interpolation=cv2.INTER_LINEAR)
        crop_start = (new_width - target_width) // 2
        resized_img_arr = resized_img_arr[:, crop_start:crop_start + target_width, :]
    else:
        # Crop the longer dimension = height
        new_height = int(target_width / asp_ratio_orig)
        resized_img_arr = cv2.resize(img_arr, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
        crop_start = (new_height - target_height) // 2
        resized_img_arr = resized_img_arr[crop_start:crop_start + target_height, :, :]

    # Transpose the image back to the original format (3 x height x width)
    return torch.from_numpy(np.ascontiguousarray(np.transpose(resized_img_arr, (2, 0, 1))))

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
    elif dataset_name == "fgvc":
        dataset = fgvcdata.Aircraft
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

def save_dataset_to_hdf5(dataset, data_out_path, resize = False, rsz_width = 32, rsz_height = 32, bot_cut = None):
    """Function to save given datset to a path as an h5 file. Option to resize the images if required.
    Args:
        dataset (torch dataset): Torch dataset object of data to save to h5 dataset.
        data_out_path (string): string of location to save h5 dataset to
        resize (bool, optional): Whether to resize the images. Defaults to False.
        rsz_width (int, optional): If resize, the target images' width. Defaults to 32.
        rsz_height (int, optional): If resize, the target images' height. Defaults to 32.
        bot_cut (int or None, optional): Some images have copyright info on the bottom of the image to remove. If not None, the number of pixels to cut from the bottom. Defaults to None.
    """
    with h5py.File(data_out_path, 'w') as hdf5_file:
        # Create a dataset within the HDF5 file to store the x data
        if resize:
            img_shape = (3, rsz_height, rsz_width)
        else:
            img_shape = dataset[0][0].shape
        x_data = hdf5_file.create_dataset('x_data', shape=(len(dataset),) + img_shape, dtype='float32')
        y_data = hdf5_file.create_dataset('y_data', shape=(len(dataset),), dtype='int8')
        # Iterate over the dataset and write x data to the HDF5 file
        for i, (x, y) in enumerate(dataset):
            # space to alter the images if required
            if bot_cut is not None: # Function to cut the bottom k pixels from the image. If not None, this will be a number of pixels.
                x = x[:, :-bot_cut, :]
            if resize: # Resize to the given width and height, using function for constant aspect ratio & cropping
                x = img_resize_const_ratio(x, rsz_width, rsz_height)
            x_data[i] = x.numpy()
            y_data[i] = y
    print(f"Dataset saved to {data_out_path}")

def ld_batch(hdf5_file, indices):
    x_data = hdf5_file["x_data"]
    y_data = hdf5_file["y_data"]
    batch_x = x_data[indices]
    batch_y = y_data[indices]
    return torch.from_numpy(batch_x), torch.tensor(batch_y)

def load_h5_batches(data_in_path, batch_size, return_single_batch = False, single_batch_index = None):
    """Function to load in image data batches from h5 file. Defaults to loading them all in and yielding a generator that can iterate through the batches (without loading all into memory).
    Alternative option to only load in a single batch, with it's index given, if wanting to distribute the processing rather than handling all batches from single process.
    Assumption that batch_size is a divisor of the length of the dataset, not yet tested what happens if not.
    Args:
        data_in_path (str): path for the input .h5 data
        batch_size (int): size of batches to be return
        return_single_batch (bool, optional): whether the function should return one batch of the data, or an interator of all batch. Defaults to False.
        single_batch_index (int, optional): if return_single_batch selected, index which batch is desired. Defaults to None.
    Returns:
        torch tensor: If return_single_batch, then this will be a tuple of a tensor of a single batch of images and of their labels
    Yields:
        iterator of torch tensors: If not return_single_batch, this will be an iterator of all data
    """
    def yield_generator(): # Need to do this, as python gets confused if both yield and return ar in the same function
        yield batch_x, batch_y

    with h5py.File(data_in_path, 'r') as hdf5_file:
        if return_single_batch:
            assert (type(single_batch_index) is int), "For returning a single batch, need single_batch_index to be an integer of which batch to select"
            batch_indices = [(batch_size * single_batch_index) + i for i in range(batch_size)]
            batch_x, batch_y = ld_batch(hdf5_file, batch_indices)
            return (batch_x, batch_y)
        else: # iterate to return all data
            x_data = hdf5_file["x_data"] # if we need this to get the length of the dataset
            # Generate indices for batches
            total_samples = len(x_data)
            indices = np.arange(total_samples)
            # Split indices into batches
            # batch_indices = [indices[i:i + batch_size] for i in range(0, total_samples, batch_size)]
            for index in indices:
                batch_x, batch_y = ld_batch(hdf5_file, index)
                return yield_generator # yield instead of return gives the batches as a generator, to avoid loading all the data into menory

def igpt_model_setup(device_name, pretrained_ref = 'openai/imagegpt-small'):
    # Open the ImageGPT model from torchvision, save it and set up CPU/GPU device
    feature_extractor = ImageGPTImageProcessor.from_pretrained(pretrained_ref)
    model = ImageGPTForCausalImageModeling.from_pretrained(pretrained_ref)
    clusters = feature_extractor.clusters
    device = torch.device(device_name)
    model.to(device)
    return feature_extractor, model, clusters, device

def demo_image(train_dataset, dataset_name = "cifar10"):
    """ If we want to plot a demo image from the data.
        If CIFAR data, use the bird from DSTL's presentation (training image 2993)
    Args:
        train_dataset (torch dataset):training dataset
        dataset_name (string, optional): The name of the dataset to import. Currently accepting "cifar10" or "fgvc". Defaults to "cifar10".
    """
    # Create a sample RGB image tensor
    if dataset_name == "cifar10":
        sample_image = train_dataset[2993][0]
    else:
        sample_image = train_dataset[0][0]
    # Convert the tensor to a NumPy array
    image_np = sample_image.numpy()
    # Transpose the NumPy array to have shape [32, 32, 3] for proper visualization
    image_np = image_np.transpose(1, 2, 0)
    # Display the image using matplotlib
    plt.imshow(image_np)
    plt.axis('off')  # Turn off the axis labels
    plt.show()

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

if __name__ == "__main__":
    # Predefine docopt options for command line interface
    __doc__ = """Generate synthetic images, using image-gpt from hugging-face

    Usage:
    igpt_generator_for_isca.py [options]

    Options:
    -h --help                   Show this screen.
    --out_path=<output fldr>    Location to save the output images to [default: data/synthetic_cifar]
    --dataset_name=<set str>    The name of the dataset to import. Currently accepting "cifar10" or "fgvc". [default: cifar10]
    --multi=<m_process bool>    Whether to run in single or multi-process. Boolean True/False [default: False]
    --h5_path=<file str>        If multi-processing, set location to store intermediate cifar data. [default: cifar_train_samp]
    --samples=<N>               Number of samples [default: 1024]
    --img_size=<1D/2Dlist int>  Size of output image. Either 1D int for square image, or 2D list of int for rectangular [default: 32]    
    --rep=<rep sizes>           Replicate sizes. Comma separated list. [default: 2]
    --prompt=<prompt sizes>     Prompt sizes. Comma separated list of {small,med,large} [default: med]
    --seed_list=<samp seeds>    Seed(s) to be used for sampling from CIFAR data. Comma separated list of integers. If multiple provided, will compute one set of scores per seed. [default: 42]
    --save=<save data>          Whether to save the images to specified folder. Boolean True/False [default: True]
    --demo=<demo>               Run a simple demo of 2 sets of images (a bird and a truck). Boolean True/False [default: False]
    --batch=<batch size>        The number of batches to process at a time. Only tested for batch size as a divisor of the number samples. Int [default: 64]
    --batch_ind=<batch_index>   If multi-processing on ISCA using job arrays, then only want this script to run for one batch, so use this index. Int [default: 0]
    --device=<device>           The name of the device to be used [default: cpu]
    """

    args = docopt(__doc__)

    out_path = args['--out_path']
    dataset_name = args['--dataset_name']
    multi_process = True if args['--multi'] == "True" else False
    h5_path = args['--h5_path']
    n_samp = int(args['--samples'])
    out_img_sz = [int(n) for n in args['--img_size'].split(',')]
    out_img_sz = out_img_sz[0] if len(out_img_sz) == 1 else out_img_sz # if single entry, convert to single int instead of len-1 list
    rep_sizes = [int(r) for r in args['--rep'].split(',')]
    prompts = args['--prompt'].split(',')
    seed_list = [int(s) for s in args['--seed_list'].split(',')]
    save_images = True if args['--save'] == "True" else False
    demo = True if args['--demo'] == "True" else False
    batch_size = int(args['--batch'])
    batch_ind = int(args['--batch_ind'])
    device_name = args['--device']
       
    # out_path = "data/synthetic_fgvc/multi_test"
    # dataset_name = "fgvc"
    # multi_process = True
    # h5_path = "fgvc_resized_train_samp_1024"
    # n_samp = 1024
    # rep_sizes = [2, 4, 6, 8]
    # seed_list = [42]
    # prompts = ["small","med","large"]
    # # save_images = False ##########
    # demo = False
    # batch_size = 64
    # batch_ind = 0 if multi_process else None
    # device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    # device_name = "cpu"


    #Define other key parameters/code running options
    test_data_needed = False # saves downloading a chunk of data
    # out_img_sz = 32 # size of the cifar image (assume square)
    # out_img_sz = [480, 800]
    igpt_temperature = 1.0 # Controls the variability of output images, default is 1
    #rep_sizes = [2, 4, 6, 8] # number of synthetic replicates (as used in DSTL study)
    prompt_defs = {'small': 0.25, 'med': 0.5, 'large': 0.75} #proportion of original image to be used as prompt (specifically the prompt is the top x % rows of the image)
    prompt_sizes = {}
    for p, v in prompt_defs.items():
        if p in prompts:
            prompt_sizes[p] = v

    print('Samples:', n_samp)
    print('Replicates:', rep_sizes)
    print('Prompts:', prompt_sizes)
    print('Output Path:', out_path)
    print('Seed(s) list:', seed_list)
    print('Batch size:', batch_size)
    print('Saving images?:', save_images)

    if save_images: # set up directories & paths for where to save images
        import cv2
        import string
        if demo:
            out_path = out_path + '_demo_data'
        try:
            os.mkdir(out_path) # create directory if it doesn't exist
        except FileExistsError:
            pass # no problem

    # Get the required components of the ImageGPT model and set it up on the device
    feature_extractor, model, clusters, device = igpt_model_setup(device_name)
    
    if not multi_process:
        train_dataset, _ = import_data_to_mem(dataset_name)

    if demo: # will only run once for demo
        demo_image(train_dataset)
        import matplotlib.pyplot as plt
        seed_list = [42]

    for seed in seed_list:
        this_h5_path = h5_path + f"_seed-{seed}.h5"

        if multi_process:
            ### need the h5 file to have already been created, haven't done a check for that yet
            x_batch, y_batch = load_h5_batches(this_h5_path, batch_size, return_single_batch = True, single_batch_index = batch_ind)
            iterator = zip(x_batch, y_batch)
        else:
            if not demo: # If running a demo, replace the training sampled data with a two-image subset, specifying image 2993 for the bird used in DSTL presentation
                train_samp = utils.data_sampler(train_dataset, n_samp, rand_seed = seed)
            else:
                train_samp = utils.data_sampler(train_dataset, 2, specific_ind = [2993], rand_seed = seed)
            save_dataset_to_hdf5(train_samp, this_h5_path)
            iterator = load_h5_batches(this_h5_path, batch_size, return_single_batch = False)

        for index, (raw_img, label) in enumerate(iterator): # Loop through the images to generate from
            if multi_process:
                random.seed(19970 + (batch_size * batch_ind) + index) # when saving the images, ensure that the random string names don't accidentally replicate!
            # print(raw_img.shape, label)
            print(f"Generating from image {index+1} of {batch_size}, for batch {batch_ind} and seed {seed}.")
            if type(out_img_sz) == int: # Assume image is square
                img_sz_y = out_img_sz
                img_sz_x = out_img_sz
            elif len(out_img_sz) == 2: # Assume rectangular
                img_sz_y = out_img_sz[0]
                img_sz_x = out_img_sz[1]
            else:
                raise ValueError(f"Unrecognised outpuut image size {out_img_sz}")

            # Create images for the given set of replicate sizes, and the given prompt sizes
            new_samples_img = create_igpt_img(raw_img, out_img_sz, feature_extractor, clusters, model, device, rep_sizes=rep_sizes, prompt_sizes=prompt_sizes, 
                                               igpt_temp=igpt_temperature, save_images=save_images, demo=demo, out_path=out_path, label=label, seed=seed)
    print("Finished!")
