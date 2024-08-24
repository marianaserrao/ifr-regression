import argparse
import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .EfficientUNetPlusPlus.utils.dataset import CoronaryArterySegmentationDataset, RetinaSegmentationDataset
from .EfficientUNetPlusPlus.segmentation_models_pytorch import segmentation_models_pytorch as smp

from torch.backends import cudnn

class VesselSegmentation:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = CoronaryArterySegmentationDataset
        self.model = self._load_model("/home/guests/mas/angiograms/enhancedModel.pth")

    def _load_model(self, path):
        print("Loading model {}".format(path))
        net = smp.EfficientUnetPlusPlus(encoder_name="timm-efficientnet-b5", encoder_weights="imagenet", in_channels=3, classes=3)
        net = nn.DataParallel(net)
        net.to(device=self.device)
        # faster convolutions, but more memory
        cudnn.benchmark = True
        net.load_state_dict(torch.load(path, map_location=self.device))
        print("Model loaded !")

        return net

    def read_image(self, path):
        image = Image.open(path).convert(mode='RGB')
        return image
    
    def save_mask(self, mask, path):
        image = self.dataset.mask2image(mask)
        image.save(path)

    def predict_mask(self, full_img, scale_factor=1, n_classes=3):
        self.model.eval()

        img = torch.from_numpy(self.dataset.preprocess(full_img, scale_factor))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img)

            if n_classes > 1:
                probs = torch.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(full_img.size[1]),
                    transforms.ToTensor()
                ]
            )

            full_mask = tf(probs.cpu())   
        
        mask = self.dataset.one_hot2mask(full_mask)
        return mask

def test():
    vessel_seg = VesselSegmentation()
    image_path = "/media/jlsstorage/masstorage/angiograms/Videos/780/5/780_29.8_24.6_27_IM-0224-0077.png"
    image = vessel_seg.read_image(image_path)
    mask = vessel_seg.predict_mask(image)
    print(mask.shape)

if __name__ == "__main__":
    test()