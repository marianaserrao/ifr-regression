import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates


# Custom elastic deformation function
def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    # If the image is 2D (grayscale), add a channel dimension
    if image.ndim == 2:
        image = image[np.newaxis, ...]

    shape = image.shape  # Shape is (channels, height, width)
    channels, height, width = shape

    # Generate random displacements
    dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Generate grid for indices
    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = np.reshape(x + dx, (-1,)), np.reshape(y + dy, (-1,))

    # Apply the same distortion for each channel
    distorted_image = np.zeros_like(image)
    for i in range(channels):
        distorted_image[i] = map_coordinates(image[i], [indices[0], indices[1]], order=1, mode='reflect').reshape(height, width)

    return distorted_image


class ElasticTransform:
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img, random_state=None):
        img = np.array(img)

        # If the image is grayscale, add a channel dimension
        if img.ndim == 2:
            img = img[np.newaxis, ...]

        if img.ndim == 3:  # If the image has multiple channels (RGB or single-channel)
            img = elastic_transform(img, self.alpha, self.sigma, random_state)
        
        return torch.from_numpy(img)


# Function to get random parameters for transformations
def get_random_transform_params(img_size):
    flip_horizontal = random.random() > 0.5
    flip_vertical = random.random() > 0.5
    rotation_angle = random.uniform(-15, 15)
    translate_x = random.uniform(-0.1, 0.1) * img_size[0]
    translate_y = random.uniform(-0.1, 0.1) * img_size[1]
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.1, 0.1)
    elastic_random_state = np.random.RandomState(None)

    return {
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "rotation_angle": rotation_angle,
        "translate": (translate_x, translate_y),
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "hue": hue,
        "elastic_random_state": elastic_random_state,
    }


def get_context_frame_ids(exam, config):
    kf_id = int(exam["key_frame"]["path"].split('_')[3])
    frame_ids = np.arange(max(1, kf_id - config.frame.window + 1), kf_id + 1, config.frame.step)

    total_frames = config.frame.window // config.frame.step
    if len(frame_ids) < total_frames:
        difference = total_frames - len(frame_ids)
        half_diff = difference // 2

        padding_i = np.ones(half_diff, dtype=int)
        padding_f = np.full(difference - half_diff, frame_ids[-1], dtype=int)

        frame_ids = np.concatenate((padding_i, frame_ids, padding_f))
    return frame_ids


def get_frame_path_by_id(key_frame_path, id):
    dirs = "/".join(key_frame_path.split('/')[:-1])
    kf_name = key_frame_path.split('/')[-1]
    frame_name = '_'.join(kf_name.split('_')[:3] + [str(id)] + kf_name.split('_')[4:])
    return os.path.join(dirs, frame_name)


class Dataset_CRNN(data.Dataset):
    def __init__(self, exams, labels, config, num_augmentations=1):
        self.labels = labels
        self.exams = exams
        self.config = config
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.exams) * self.num_augmentations

    def read_images(self, exam, transform, random_params=None):
        X = []
        kf_ids = get_context_frame_ids(exam, self.config)
        key_frame_path = exam["key_frame"]["path"]

        for i in kf_ids:
            image = Image.open(get_frame_path_by_id(key_frame_path, i))

            # Apply the transform
            if random_params:
                image = transform(image, random_params)
                # image = transform(image)
            else:
                image = transform(image)

            X.append(image)

        # Load the key frame mask and apply transformation
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")
        kf_mask = Image.merge("RGB", (kf_mask, kf_mask, kf_mask))  # Convert to 3-channel mask
        # Apply the transform
        if random_params:
            kf_mask = transform(kf_mask, random_params)
        else:
            kf_mask = transform(kf_mask)
        X.extend([kf_mask] * self.config.frame.n_mask)

        # Stack frames into a tensor and adjust based on patient's age
        X = torch.stack(X, dim=0)
        age_factor = exam["patient"]["age"] or 100
        X = X * age_factor / 100

        return X

    def __getitem__(self, index):
        exam_index = index // self.num_augmentations
        augment_index = index % self.num_augmentations

        exam = self.exams[exam_index]

        if augment_index == 0:
            # Base transform
            X = self.read_images(exam, transform=self.get_base_transform())
        else:
            # Augmentation transform
            random_params = get_random_transform_params((self.config.crnn.cnn.in_dim, self.config.crnn.cnn.in_dim))
            X = self.read_images(exam, transform=self.apply_augment, random_params=random_params)

        y = torch.FloatTensor([self.labels[exam_index]])
        return X, y

    def get_base_transform(self):
        return transforms.Compose([
            transforms.Resize([self.config.crnn.cnn.in_dim, self.config.crnn.cnn.in_dim]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_augment(self, image, params):
        # Apply the random params consistently across all frames
        image = F.resize(image, [self.config.crnn.cnn.in_dim, self.config.crnn.cnn.in_dim])
        if params["flip_horizontal"]:
            image = F.hflip(image)
        # if params["flip_vertical"]:
        #     image = F.vflip(image)
        image = F.affine(image, angle=params["rotation_angle"], translate=params["translate"], scale=1.0, shear=0)
        image = F.adjust_brightness(image, params["brightness"])
        image = F.adjust_contrast(image, params["contrast"])
        image = F.adjust_saturation(image, params["saturation"])
        image = F.adjust_hue(image, params["hue"])

        image = F.to_tensor(image)

        elastic_transformer = ElasticTransform(alpha=34, sigma=4)
        image = elastic_transformer(image, random_state=params["elastic_random_state"])

        image=F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image
