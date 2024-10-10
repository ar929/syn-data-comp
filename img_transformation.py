import random
import warnings
import numpy as np
import torch
from torchvision import transforms

def img_transformation(data, prob=1, gamma=3, random_erase_greyscale=True, seed=None):
    """Function to apply transformations to an image or batch of images.

    Args:
        data (tensor): A 3D image (single) or a 4D tensor (batch).
        prob (float, 0 <= prob <= 1): Probability that any given image will be transformed. Defaults to 0.5.
        gamma (float): Transformation intensity parameter. Clamped between 0 and 10, defaults to 3.
        random_erase_greyscale (bool, optional): Whether to include random erasure and grayscale transformations. Defaults to True.
        seed (int, optional): A seed for reproducibility. Defaults to None.
    """

    # Clamp gamma to be between 0 and 10
    gamma = min(max(gamma, 0), 10)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Define common transformations
    transform_list = [
        transforms.RandomResizedCrop(32, (1 - 0.1 * gamma, 1 + 0.2 * gamma), antialias=True),
        transforms.Pad(padding=12, fill=0, padding_mode='symmetric'),
        transforms.RandomAffine(degrees=9 * gamma, shear=3.3 * gamma),
        transforms.CenterCrop(32),
        transforms.ColorJitter(brightness=0.08 * gamma, contrast=0.08 * gamma, saturation=0.1 * gamma, hue=(0.06 * gamma if gamma <= 5 else 0.5))
    ]
    
    # Add conditional transformations for higher gamma values
    if gamma >= 1:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.GaussianBlur(kernel_size=int(np.floor(gamma / 4)) * 2 + 1))
    
    if random_erase_greyscale:
        transform_list.append(transforms.RandomGrayscale(p=0.04 * gamma))
        if gamma >= 1:
            transform_list.append(transforms.RandomErasing(p=(0.15 * gamma if gamma <= 6 else 1), scale=(0.01 * gamma, 0.05 * gamma)))

    transform = transforms.Compose(transform_list)

    # Apply transformations to the data
    if data.ndim == 3:  # Single image
        return transform(data) if random.random() < prob else data
    elif data.ndim == 4:  # Batch of images
        out_imgs = torch.empty_like(data)
        for i in range(len(data)):
            out_imgs[i] = transform(data[i]) if random.random() < prob else data[i]
        return out_imgs
    else:
        raise ValueError("Input tensor must be 3D (single image) or 4D (batch of images).")
