from preprocess.patient_data import get_patient_data
from .utils import *
from .models import *
from .dataset import *
from .transforms import *
from .crnn import train, test

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import json
import torch.multiprocessing as mp

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
    with open(full_config.patient_json_path, 'r') as file:
        patient_data = json.load(file)

    with open(full_config.exam_json_path, 'r') as file:
        clinical_data = json.load(file)

    patient_ids = list(patient_data.keys())
    train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=config.test_size, random_state=SEED)

    X_train = [exam for exam in clinical_data if exam["patient_id"] in train_patient_ids]
    y_train = [patient_data[exam["patient_id"]]["ifr"] for exam in X_train]
    X_test = [exam for exam in clinical_data if exam["patient_id"] in test_patient_ids]
    y_test = [patient_data[exam["patient_id"]]["ifr"] for exam in X_test]

    transform = get_crnn_transform(config.crnn.cnn.in_dim,config.crnn.cnn.in_dim)
    aug_transform = get_crnn_augmentation_tranform(config.crnn.cnn.in_dim,config.crnn.cnn.in_dim)

    train_set = Dataset_CRNN(X_train, y_train, config, transform=aug_transform)
    test_set = Dataset_CRNN(X_test, y_test, config, transform=transform)

    train_loader = data.DataLoader(train_set, **params)
    test_loader = data.DataLoader(test_set, **params)

    # # # create model
    # # cnn_config = config.crnn.cnn
    # # cnn_encoder = ResNetCNNEncoder(
    # #     fc1_dim=cnn_config.fc1_dim, 
    # #     fc2_dim=cnn_config.fc2_dim, 
    # #     drop_p=cnn_config.dropout_p, 
    # #     out_dim=cnn_config.out_dim,
    # # ).to(device)

    # create model
    cnn_config = config.crnn.cnn
    cnn_encoder = VGGCNNEncoder(
        fc1_dim=cnn_config.fc1_dim, 
        fc2_dim=cnn_config.fc2_dim, 
        drop_p=cnn_config.dropout_p, 
        out_dim=cnn_config.out_dim,
    ).to(device)

    rnn_config = config.crnn.rnn
    rnn_decoder = RNNDecoder(
        in_dim=cnn_config.out_dim, 
        hidden_layers=rnn_config.hidden_layers, 
        hidden_size=rnn_config.hidden_size, 
        fc1_dim=rnn_config.fc1_dim, 
        drop_p=rnn_config.dropout_p
    ).to(device)

    # parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cnn_encoder = nn.DataParallel(cnn_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)
        # combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                    list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                    list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

    elif torch.cuda.device_count() == 1:
        print("Using", torch.cuda.device_count(), "GPU!")
        # combine all EncoderCNN + DecoderRNN parameters
        crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                    list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                    list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=config.crnn.lr, weight_decay=config.crnn.weight_decay)

    # record training process
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    # start training
    for epoch in range(config.epochs):
        # train, test model
        train_losses, train_scores = train(config.log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch, config)
        epoch_test_loss, epoch_test_score = test([cnn_encoder, rnn_decoder], device, optimizer, test_loader, epoch, config)

        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        # save performace files
        if config.save_checkpoints:
            np.save(os.path.join(config.performance_dir,'crnn_epoch_training_losses.npy'), np.array(epoch_train_losses))
            np.save(os.path.join(config.performance_dir,'crnn_epoch_training_scores.npy'), np.array(epoch_train_scores))
            np.save(os.path.join(config.performance_dir,'crnn_epoch_test_loss.npy'), np.array(epoch_test_losses))
            np.save(os.path.join(config.performance_dir,'crnn_epoch_test_score.npy'), np.array(epoch_test_scores))

            # Save the model checkpoint
            torch.save({
                'epoch': epoch,
                'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                'rnn_decoder_state_dict': rnn_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'test_loss': epoch_test_loss,
                'train_score': train_scores,
                'test_score': epoch_test_score,
            }, os.path.join(config.checkpoints_dir, f'{config.model_name}_checkpoint_epoch_{epoch}.pth'))
    
    # plot_performance("ResNetRCNN", config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores)
    plot_performance(config.model_name, config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores)

if __name__ =="__main__":
    main()