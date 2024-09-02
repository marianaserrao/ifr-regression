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
        loss = F.binary_cross_entropy(output, y)
        losses.append(loss.item())

        y_pred = output > 0.5

        # Compute accuracy
        accuracy = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())    

        # compute step_score  
        step_score = accuracy
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
            loss = F.binary_cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()   # sum up batch loss

            y_pred = output>0.5

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            
    # compute average test loss
    test_loss /= len(test_loader.dataset)

    # Convert lists to numpy arrays for metric calculations
    all_y = torch.stack(all_y, dim=0).cpu().data.squeeze().numpy()
    all_y_pred = torch.stack(all_y_pred, dim=0).cpu().data.squeeze().numpy()

    # Compute binary class performance
    accuracy = accuracy_score(all_y, all_y_pred)
    tn, fp, fn, tp = confusion_matrix(all_y, all_y_pred).ravel()
    sensitivity = recall_score(all_y, all_y_pred)
    specificity = tn / (tn + fp)
    ppv = precision_score(all_y, all_y_pred)
    npv = tn / (tn + fn)

    # compute test_score  
    test_score = accuracy

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