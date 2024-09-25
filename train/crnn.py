import os
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report, precision_score, recall_score
import numpy as np

def train(log_interval, model, device, train_loader, optimizer, epoch, config):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    sample_counter = 0   # total trained sample in one epoch

    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, 1)

        sample_counter += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # shape = (batch, 1)

        # compute loss
        loss = F.mse_loss(output, y)
        losses.append(loss.item())

        # compute mse on cpu
        y_pred = output

        # Check for NaN values in predictions or ground truth
        if torch.isnan(y_pred).any() or torch.isnan(y).any():
            print("NaN detected in predictions or labels")
            print(f"y_pred: {y_pred}")
            print(f"y: {y}")
            raise ValueError("NaN detected in predictions or labels")
        for param in cnn_encoder.parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in encoder parameters")
        for param in rnn_decoder.parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in decoder parameters")
        if torch.isnan(X).any():
            print("NaN detected in input data")

        mse = mean_squared_error(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        # compute binary class performance
        thr = 0.89
        bin_y = np.where(y.cpu().data.squeeze().numpy() < thr, 1, 0)
        bin_y_pred = np.where(y_pred.cpu().data.squeeze().numpy() < thr, 1, 0)
        # Compute accuracy
        accuracy = accuracy_score(bin_y, bin_y_pred)    

        # compute step_score  
        step_score = mse if config.metric=="MSE" else accuracy
        scores.append(step_score)   

        # update weights
        loss.backward()
        optimizer.step()

        # log results
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, {}: {:.6f}'.format(
                epoch + 1, 
                sample_counter, 
                len(train_loader.dataset), 
                100. * (batch_idx + 1) / len(train_loader), 
                loss.item(), 
                config.metric,
                step_score
            ))

    return losses, scores

def test(model, device, optimizer, test_loader, epoch, config):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, 1)

            output = rnn_decoder(cnn_encoder(X))

            # compute loss
            loss = F.mse_loss(output, y, reduction='sum')
            test_loss += loss.item()   # sum up batch loss
            y_pred = output

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            
    # compute average test loss
    test_loss /= len(test_loader.dataset)

    # compute MSE on CPU
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    mse = mean_squared_error(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # compute binary class performance
    thr = 0.89
    bin_y = np.where(all_y.cpu().data.squeeze().numpy() < thr, 1, 0)
    bin_y_pred = np.where(all_y_pred.cpu().data.squeeze().numpy() < thr, 1, 0)
    # Compute accuracy
    accuracy = accuracy_score(bin_y, bin_y_pred)
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(bin_y, bin_y_pred).ravel()
    # Compute sensitivity (recall)
    sensitivity = recall_score(bin_y, bin_y_pred)
    # Compute specificity
    specificity = tn / (tn + fp)
    # Compute positive predictive value (PPV)
    ppv = precision_score(bin_y, bin_y_pred)
    # Compute negative predictive value (NPV)
    npv = tn / (tn + fn)

    # compute test_score  
    test_score = mse if config.metric=="MSE" else accuracy

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, {}: {:.4f}\n'.format(len(all_y), test_loss, config.metric, test_score))
    # Display the metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Positive Predictive Value (PPV): {ppv:.2f}")
    print(f"Negative Predictive Value (NPV): {npv:.2f}")
    print()

    # save checkpoints
    if config.save_checkpoints:
        torch.save(cnn_encoder.state_dict(), os.path.join(config.checkpoints_dir, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        torch.save(rnn_decoder.state_dict(), os.path.join(config.checkpoints_dir, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save temporal_decoder
        torch.save(optimizer.state_dict(), os.path.join(config.checkpoints_dir, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score