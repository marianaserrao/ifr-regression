import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, exams, labels, config, transform=None):
        "Initialization"
        self.labels = labels
        self.exams = exams
        self.config = config.crnn
        self.frame_window = config.frame.window
        self.frame_step = config.frame.step
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.exams)
    
    def get_context_frame_ids(self, exam):
        kf_id = exam["key_frame"]["id"]
        frame_ids = np.arange(max(1, kf_id-self.frame_window*self.frame_step), kf_id, self.frame_step)

        if len(frame_ids) < self.frame_window:
            difference = self.frame_window - len(frame_ids)
            half_diff = difference // 2
            
            # Create the padding arrays
            padding_i = np.ones(half_diff, dtype=int)
            padding_f = np.full(difference - half_diff, frame_ids[-1], dtype=int)
            
            # Concatenate the numbers with the padding
            frame_ids = np.concatenate((padding_i, frame_ids, padding_f))
        return frame_ids

    def get_frame_path_by_id(self, key_frame_path, id):
        dirs = "/".join(key_frame_path.split('/')[:-1])
        kf_name = key_frame_path.split('/')[-1]
        frame_name = '_'.join(kf_name.split('_')[:3] + [str(id)] + kf_name.split('_')[4:])
        path= os.path.join(dirs, frame_name)
        return path

    def read_images(self, exam):
        X = []
        kf_ids=self.get_context_frame_ids(exam)
        key_frame_path = exam["key_frame"]["path"]
        
        for i in kf_ids:            
            image = Image.open(self.get_frame_path_by_id(key_frame_path, i))
            # image = Image.merge("RGB", (image, image, image))

            if self.transform is not None:
                image = self.transform(image)

            X.append(image)
        
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")
        kf_mask = Image.merge("RGB", (kf_mask, kf_mask, kf_mask))
        if self.transform is not None:
            kf_mask = self.transform(kf_mask)
        X.append(kf_mask)

        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        exam = self.exams[index]

        # Load data
        X = self.read_images(exam)     # (input) spatial images
        y = torch.FloatTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        return X, y
    
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, exams, labels, config, transform=None):
        "Initialization"
        self.labels = labels
        self.exams = exams
        self.config = config.cnn3d
        self.frame_window = config.frame.window
        self.frame_step = config.frame.step
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.exams)
    
    def get_context_frame_ids(self, exam):
        kf_id = exam["key_frame"]["id"]
        frame_ids = np.arange(max(1, kf_id-self.frame_window*self.frame_step), kf_id, self.frame_step)

        if len(frame_ids) < self.frame_window:
            difference = self.frame_window - len(frame_ids)
            half_diff = difference // 2
            
            # Create the padding arrays
            padding_i = np.ones(half_diff, dtype=int)
            padding_f = np.full(difference - half_diff, frame_ids[-1], dtype=int)
            
            # Concatenate the numbers with the padding
            frame_ids = np.concatenate((padding_i, frame_ids, padding_f))
        return frame_ids

    def get_frame_path_by_id(self, key_frame_path, id):
        dirs = "/".join(key_frame_path.split('/')[:-1])
        kf_name = key_frame_path.split('/')[-1]
        frame_name = '_'.join(kf_name.split('_')[:3] + [str(id)] + kf_name.split('_')[4:])
        path= os.path.join(dirs, frame_name)
        return path

    def read_images(self, exam):
        X = []
        kf_ids=self.get_context_frame_ids(exam)
        key_frame_path = exam["key_frame"]["path"]
        
        for i in kf_ids:            
            image = Image.open(self.get_frame_path_by_id(key_frame_path, i)).convert("L")
            # image = Image.merge("RGB", (image, image, image))

            if self.transform is not None:
                image = self.transform(image)

            X.append(image.squeeze_(0))
        
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")
        # kf_mask = Image.merge("RGB", (kf_mask, kf_mask, kf_mask))
        if self.transform is not None:
            kf_mask = self.transform(kf_mask)
        X.append(kf_mask.squeeze_(0))

        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        exam = self.exams[index]

        # Load data
        X = self.read_images(exam).unsqueeze_(0)     # (input) spatial images
        y = torch.FloatTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        return X, y

class Dataset_2DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, exams, labels, config, transform=None):
        "Initialization"
        self.labels = labels
        self.exams = exams
        self.config = config.cnn2d
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.exams)

    def read_image(self, exam):
        
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")

        if self.transform is not None:
            kf_mask = self.transform(kf_mask)

        X = kf_mask
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        exam = self.exams[index]

        # Load data
        X = self.read_image(exam)
        y = torch.FloatTensor([self.labels[index]])
        return X, y