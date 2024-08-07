from utils import *
from models import *
from dataset import *

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import json

SEED = 42
full_config = get_config("./config.yaml")
config = full_config.training

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    sample_counter = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1,1 )

        sample_counter += X.size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        # compute loss
        loss = F.mse_loss(output, y)
        losses.append(loss.item())

        # compute mse on cpu
        y_pred = output
        step_score = mean_squared_error(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)

        # update weights
        loss.backward()
        optimizer.step()

        # log results
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, MSE: {:.6f}'.format(
                epoch + 1, 
                sample_counter, 
                len(train_loader.dataset), 
                100. * (batch_idx + 1) / len(train_loader), 
                loss.item(), 
                step_score
            ))

    return losses, scores

def validation(model, device, optimizer, test_loader, epoch):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, 1)

            output = model(X)

            # compute loss
            loss = F.mse_loss(output, y)
            test_loss += loss.item()    # sum up batch loss
            y_pred = output

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute MSE on CPU
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = mean_squared_error(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, MSE: {:.4f}\n'.format(len(all_y), test_loss, test_score))

    # save Pytorch models of best record
    if config.save_checkpoints:
        torch.save(model.state_dict(), os.path.join(config.checkpoints_dir, '3dcnn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        torch.save(optimizer.state_dict(), os.path.join(config.checkpoints_dir, '3dcnn_optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score

def main():

    # data loading parameters
    params = {
        'batch_size': config.batch_size, 
        'shuffle': True, 
        'num_workers': config.num_workers, 
        'pin_memory': True
    } if use_cuda else {}

    # load data
    with open(config.data_json, 'r') as file:
        clinical_data = json.load(file)

    clinical_data = clinical_data['kf_exams']
    clinical_data = [exam for exam in clinical_data if (exam['patient']['ifr']!=None and exam['patient']["exclude"]!=2)]

    X=clinical_data.copy()
    y=[exam['patient']['ifr'] for exam in clinical_data]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=SEED)

    train_set, valid_set = Dataset_3DCNN(X_train, y_train, config, augment=True), \
                        Dataset_3DCNN(X_test, y_test, config)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    cnn3d = CNN3D(
        t_dim=config.frame.window+config.frame.n_mask, 
        img_x=config.cnn3d.img_x,
        img_y=config.cnn3d.img_y,
        drop_p=config.cnn3d.dropout_p, 
        fc_hidden1=config.cnn3d.fc1_dim,  
        fc_hidden2=config.cnn3d.fc2_dim
    ).to(device)

    # parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn3d = nn.DataParallel(cnn3d)

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=config.cnn3d.lr)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(config.epochs):
        # train, test model
        train_losses, train_scores = train(config.log_interval, cnn3d, device, train_loader, optimizer, epoch)
        epoch_test_loss, epoch_test_score = validation(cnn3d, device, optimizer, valid_loader, epoch)

        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save performace files
        if config.save_checkpoints:
            np.save(os.path.join(config.performance_dir,'3dcnn_epoch_training_losses.npy'), np.array(epoch_train_losses))
            np.save(os.path.join(config.performance_dir,'3dcnn_epoch_training_scores.npy'), np.array(epoch_train_scores))
            np.save(os.path.join(config.performance_dir,'3dcnn_epoch_test_loss.npy'), np.array(epoch_test_losses))
            np.save(os.path.join(config.performance_dir,'3dcnn_epoch_test_score.npy'), np.array(epoch_test_scores))
    
    plot_performance("3DCNN", config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores)

if __name__ =="__main__":
    main()

