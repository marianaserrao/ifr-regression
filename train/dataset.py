import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch

from vessel_segmentation import VesselSegmentation

def get_context_frame_ids(exam, config):
    kf_id = exam["key_frame"]["id"]
    frame_ids = np.arange(max(1, kf_id-config.frame.window+1), kf_id+1, config.frame.step)

    total_frames = config.frame.window//config.frame.step
    if len(frame_ids) < total_frames:
        difference = total_frames - len(frame_ids)
        half_diff = difference // 2
        
        # Create the padding arrays
        padding_i = np.ones(half_diff, dtype=int)
        padding_f = np.full(difference - half_diff, frame_ids[-1], dtype=int)
        
        # Concatenate the numbers with the padding
        frame_ids = np.concatenate((padding_i, frame_ids, padding_f))
    return frame_ids

def get_frame_path_by_id(key_frame_path, id):
    dirs = "/".join(key_frame_path.split('/')[:-1])
    kf_name = key_frame_path.split('/')[-1]
    frame_name = '_'.join(kf_name.split('_')[:3] + [str(id)] + kf_name.split('_')[4:])
    path= os.path.join(dirs, frame_name)
    return path

class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, exams, labels, config, transform=None):
        "Initialization"
        self.labels = labels
        self.exams = exams
        self.config = config
        self.transform = transform
        self.vs = VesselSegmentation()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.exams)

    def read_images(self, exam):
        X = []
        kf_ids=get_context_frame_ids(exam, self.config)
        key_frame_path = exam["key_frame"]["path"]
        
        for i in kf_ids:            
            image = Image.open(get_frame_path_by_id(key_frame_path, i))

            # raw_image = self.vs.read_image(get_frame_path_by_id(key_frame_path, i))
            # mask = self.vs.predict_mask(raw_image)
            # image = Image.merge("RGB", (mask, mask, mask))

            if self.transform is not None:
                image = self.transform(image)

            X.append(image)
        
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")
        kf_mask = Image.merge("RGB", (kf_mask, kf_mask, kf_mask))
        if self.transform is not None:
            kf_mask = self.transform(kf_mask)
        X.extend([kf_mask] * self.config.frame.n_mask)

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
        self.config = config
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.exams)

    def read_images(self, exam):
        X = []
        kf_ids=get_context_frame_ids(exam, self.config)
        key_frame_path = exam["key_frame"]["path"]
        
        for i in kf_ids:            
            image = Image.open(get_frame_path_by_id(key_frame_path, i)).convert("L")
            # image = Image.merge("RGB", (image, image, image))

            if self.transform is not None:
                image = self.transform(image)

            X.append(image.squeeze_(0))
        
        kf_mask = Image.open(exam["key_frame"]["mask"]).convert("L")
        # kf_mask = Image.merge("RGB", (kf_mask, kf_mask, kf_mask))
        if self.transform is not None:
            kf_mask = self.transform(kf_mask)
        X.extend([kf_mask.squeeze_(0)] * self.config.frame.n_mask)

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
        self.config = config
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