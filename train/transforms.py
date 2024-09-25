import torchvision.transforms as transforms
import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Custom elastic deformation function
def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distorted_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted_image

class ElasticTransform:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        img = np.array(img)
        if img.ndim == 3:  # If image has multiple channels
            for i in range(img.shape[0]):
                img[i] = elastic_transform(img[i], self.alpha, self.sigma)
        else:
            img = elastic_transform(img, self.alpha, self.sigma)
        return torch.from_numpy(img)

def get_crnn_transform(img_x,img_y):    
    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Normalize(mean=[0.43, 0.43, 0.43], std=[0.15, 0.15, 0.15])
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

def get_crnn_augmentation_tranform(img_x,img_y):
    base_transform = get_crnn_transform(img_x, img_y)
    return base_transform

def get_cnn_transform(img_x,img_y):    
    transform = transforms.Compose([
        transforms.Resize([img_x, img_y]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform

def get_cnn_augmentation_tranform(img_x,img_y):
    base_transform = get_cnn_transform(img_x, img_y)
    return base_transform
