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
    clinical_data = [exam for exam in clinical_data if (exam['patient']['ifr']!=None and exam['patient']["exclude"]!=1)]

    # # correct split
    patient_ids = list(set([exam["patient"]["id"] for exam in clinical_data]))
    train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=config.test_size, random_state=SEED)

    X_train = [exam for exam in clinical_data if exam["patient"]["id"] in train_patient_ids]
    y_train = [exam["patient"]["ifr"] for exam in X_train]
    X_test = [exam for exam in clinical_data if exam["patient"]["id"] in test_patient_ids]
    y_test = [exam["patient"]["ifr"] for exam  in X_test]

    # # leak split
    # X=clinical_data.copy()
    # y=[exam['patient']['ifr'] for exam in clinical_data]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=SEED)

    transform = get_cnn_transform(config.cnn3d.img_x,config.cnn3d.img_y)
    aug_transform = get_cnn_augmentation_tranform(config.cnn3d.img_x,config.cnn3d.img_y)

    train_set = Dataset_3DCNN(X_train, y_train, config, transform=aug_transform)
    valid_set = Dataset_3DCNN(X_test, y_test, config, transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    cnn3d = CNN3D(
        t_dim=config.frame.window//config.frame.step+config.frame.n_mask, 
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

    optimizer = torch.optim.Adam(cnn3d.parameters(), lr=config.cnn3d.lr, weight_decay=config.cnn3d.weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(config.epochs):
        # train, test model
        train_losses, train_scores = train(config.log_interval, cnn3d, device, train_loader, optimizer, epoch, config)
        epoch_test_loss, epoch_test_score = validation(cnn3d, device, optimizer, valid_loader, epoch, config)

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

