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

#TODO: refactor cnn out size calc
def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

##### RCNN ##############################################################################

# 2D CNN encoder train from scratch (no transfer learning)
class CustomCNNEncoder(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(CustomCNNEncoder, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

class ResNetCNNEncoder(nn.Module):
    def __init__(self, fc1_dim=512, fc2_dim=512, drop_p=0.3, out_dim=300):
        """Load the pretrained ResNet and replace top fc layer."""
        super(ResNetCNNEncoder, self).__init__()

        self.fc1_dim, self.fc2_dim = fc1_dim, fc2_dim
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # remove the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01)
        self.fc3 = nn.Linear(fc2_dim, out_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq
    
class DenseNetCNNEncoder(nn.Module):
    def __init__(self, fc1_dim=512, fc2_dim=512, drop_p=0.3, out_dim=300):
        """Load the pretrained DenseNet and replace top fc layer."""
        super(DenseNetCNNEncoder, self).__init__()

        self.fc1_dim, self.fc2_dim = fc1_dim, fc2_dim
        self.drop_p = drop_p

        densenet = models.densenet121(pretrained=True)
        self.densenet = densenet.features  # Extract the convolutional feature extractor part
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Use adaptive pooling to flatten input
        num_features = densenet.classifier.in_features  # Get the number of input features for the classifier

        # Define fully connected layers
        self.fc1 = nn.Linear(num_features, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01)
        self.fc3 = nn.Linear(fc2_dim, out_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # DenseNet CNN
            with torch.no_grad():
                x = self.densenet(x_3d[:, t, :, :, :])  # Forward pass through DenseNet features
                x = self.avgpool(x)  # Apply adaptive pooling
                x = torch.flatten(x, 1)  # Flatten the output

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

class VGGCNNEncoder(nn.Module):
    def __init__(self, fc1_dim=512, fc2_dim=512, drop_p=0.3, out_dim=300):
        """Load the pretrained VGG and replace top fc layer."""
        super(VGGCNNEncoder, self).__init__()

        self.fc1_dim, self.fc2_dim = fc1_dim, fc2_dim
        self.drop_p = drop_p

        # Load pretrained VGG and modify it
        vgg = models.vgg16(pretrained=True)
        self.vgg = vgg.features  # Use the feature extractor part of VGG
        self.avgpool = vgg.avgpool  # VGG has an avgpool layer before the classifier
        num_features = vgg.classifier[0].in_features  # Get the number of input features for the first FC layer in VGG

        # Define fully connected layers
        self.fc1 = nn.Linear(num_features, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01)
        self.fc3 = nn.Linear(fc2_dim, out_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # VGG CNN
            with torch.no_grad():
                x = self.vgg(x_3d[:, t, :, :, :])  # Forward pass through VGG features
                x = self.avgpool(x)  # Apply VGG's avgpool
                x = torch.flatten(x, 1)  # Flatten the output
            
            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Swap time and sample dimensions to get (batch_size, time_steps, feature_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

class EfficientNetB5CNNEncoder(nn.Module):
    def __init__(self, fc1_dim=512, fc2_dim=512, drop_p=0.3, out_dim=300):
        """Load the pretrained EfficientNet-B5 and replace top fc layer."""
        super(EfficientNetB5CNNEncoder, self).__init__()

        self.fc1_dim, self.fc2_dim = fc1_dim, fc2_dim
        self.drop_p = drop_p

        # Load pretrained EfficientNet-B5 and modify it
        efficientnet = models.efficientnet_b5(pretrained=True)
        self.features = efficientnet.features  # Use the feature extractor part of EfficientNet
        self.avgpool = efficientnet.avgpool  # EfficientNet has an avgpool layer before the classifier
        num_features = efficientnet.classifier[1].in_features  # Get the number of input features for the first FC layer

        # Define fully connected layers
        self.fc1 = nn.Linear(num_features, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01)
        self.fc3 = nn.Linear(fc2_dim, out_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # EfficientNet-B5 CNN
            with torch.no_grad():
                x = self.features(x_3d[:, t, :, :, :])  # Forward pass through EfficientNet-B5 features
                x = self.avgpool(x)  # Apply EfficientNet's avgpool
                x = torch.flatten(x, 1)  # Flatten the output
            
            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Swap time and sample dimensions to get (batch_size, time_steps, feature_dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

# class InceptionV3CNNEncoder(nn.Module):
#     def __init__(self, fc1_dim=512, fc2_dim=512, drop_p=0.3, out_dim=300):
#         """Load the pretrained InceptionV3 and replace top fc layer."""
#         super(InceptionV3CNNEncoder, self).__init__()

#         self.fc1_dim, self.fc2_dim = fc1_dim, fc2_dim
#         self.drop_p = drop_p

#         # Load pretrained InceptionV3 and keep everything except the final FC layer
#         inception = models.inception_v3(pretrained=True, aux_logits=True)
#         self.inception = inception
#         num_features = inception.fc.in_features  # Get the number of input features for the classifier

#         # Define fully connected layers
#         self.fc1 = nn.Linear(num_features, fc1_dim)
#         self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01)
#         self.fc2 = nn.Linear(fc1_dim, fc2_dim)
#         self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01)
#         self.fc3 = nn.Linear(fc2_dim, out_dim)
        
#     def forward(self, x_3d):
#         cnn_embed_seq = []
#         for t in range(x_3d.size(1)):
#             # InceptionV3 CNN
#             with torch.no_grad():
#                 print(x_3d[:, t, :, :, :].shape)
#                 # Forward pass through the InceptionV3 layers
#                 x = self.inception(x_3d[:, t, :, :, :])
#                 if self.inception.aux_logits:
#                     x = x.logits  # If aux_logits is True, extract the main logits
#                 x = torch.flatten(x, 1)  # Flatten the output
            
#             # FC layers
#             x = self.bn1(self.fc1(x))
#             x = F.relu(x)
#             x = self.bn2(self.fc2(x))
#             x = F.relu(x)
#             x = F.dropout(x, p=self.drop_p, training=self.training)
#             x = self.fc3(x)

#             cnn_embed_seq.append(x)

#         # Swap time and sample dimensions to get (batch_size, time_steps, feature_dim)
#         cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

#         return cnn_embed_seq

class RNNDecoder(nn.Module):
    def __init__(self, in_dim=300, hidden_layers=3, hidden_size=256, fc1_dim=128, drop_p=0.3, num_classes=1):
        super(RNNDecoder, self).__init__()

        self.RNN_input_size = in_dim
        self.hidden_layers = hidden_layers   # RNN hidden layers
        self.hidden_size = hidden_size                 # RNN hidden nodes
        self.fc1_dim = fc1_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.hidden_size,        
            num_layers=hidden_layers,       
            batch_first=True,       # input & output will have batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.hidden_size, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.num_classes)

    def forward(self, x_RNN):        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

##### 3D CNN #######################################################

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=1):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x

##### 2D CNN ############################################################################
class CNN2D(nn.Module):
    def __init__(self, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=1):
        super(CNN2D, self).__init__()

        # set image dimension
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5), (3, 3)  # 2d kernel size
        self.s1, self.s2 = (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2 = (0, 0), (0, 0)  # 2d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm2d(self.ch1)
        self.conv2 = nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm2d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1], self.fc_hidden1)  # fully connected hidden layer
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = regression value

    def forward(self, x_2d):
        # Conv 1
        x = self.conv1(x_2d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x