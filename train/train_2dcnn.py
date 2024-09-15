from .utils import *
from .models import *
from .dataset import *
from .transforms import *
from .cnn import train, validation

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import json

def main():

    SEED = 42
    full_config = get_config("./config.yaml")
    config = full_config.training

    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

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

    transform = get_cnn_transform(config.cnn2d.img_x,config.cnn2d.img_y)
    aug_transform = get_cnn_augmentation_tranform(config.cnn2d.img_x,config.cnn2d.img_y)

    train_set = Dataset_2DCNN(X_train, y_train, config, transform=aug_transform)
    valid_set = Dataset_2DCNN(X_test, y_test, config, transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    cnn2d = CNN2D(
        img_x=config.cnn2d.img_x,
        img_y=config.cnn2d.img_y,
        drop_p=config.cnn2d.dropout_p, 
        fc_hidden1=config.cnn2d.fc1_dim,  
        fc_hidden2=config.cnn2d.fc2_dim
    ).to(device)

    # parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn2d = nn.DataParallel(cnn2d)

    optimizer = torch.optim.Adam(cnn2d.parameters(), lr=config.cnn2d.lr, weight_decay=config.cnn2d.weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(config.epochs):
        # train, test model
        train_losses, train_scores = train(config.log_interval, cnn2d, device, train_loader, optimizer, epoch, config)
        epoch_test_loss, epoch_test_score = validation(cnn2d, device, optimizer, valid_loader, epoch, config)

        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save performace files
        if config.save_checkpoints:
            np.save(os.path.join(config.performance_dir,'2dcnn_epoch_training_losses.npy'), np.array(epoch_train_losses))
            np.save(os.path.join(config.performance_dir,'2dcnn_epoch_training_scores.npy'), np.array(epoch_train_scores))
            np.save(os.path.join(config.performance_dir,'2dcnn_epoch_test_loss.npy'), np.array(epoch_test_losses))
            np.save(os.path.join(config.performance_dir,'2dcnn_epoch_test_score.npy'), np.array(epoch_test_scores))
    
    plot_performance("2DCNN", config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores)

if __name__ =="__main__":
    main()

