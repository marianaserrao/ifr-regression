from box import Box
from matplotlib import pyplot as plt
import numpy as np
import yaml
import os
import time

def get_config(file_path):
    """Load a YAML file and return a Box object (dot notation)"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return Box(data)

def plot_performance(model_name, config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores):
    fig = plt.figure(figsize=(10, 4))

    epoch_train_scores, epoch_train_losses = np.array(epoch_train_scores)[:, -1], np.array(epoch_train_losses)[:, -1]
    epoch_test_scores, epoch_test_losses = np.array(epoch_test_scores), np.array(epoch_test_losses)

    # plot losses
    plt.subplot(121)
    plt.plot(np.arange(1, config.epochs + 1), epoch_train_losses)  # train loss (on epoch end)
    plt.plot(np.arange(1, config.epochs + 1), epoch_test_losses)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")

    # plot mse
    plt.subplot(122)
    plt.plot(np.arange(1, config.epochs + 1), epoch_train_scores)  # train accuracy (on epoch end)
    plt.plot(np.arange(1, config.epochs + 1), epoch_test_scores)         #  test accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel(config.metric)
    plt.legend(['train', 'test'], loc="upper left")

    path = os.path.join(config.charts_dir, f"{model_name}_{int(time.time())}.png")
    plt.savefig(path, dpi=600)
    plt.close(fig)

    get_best_score = np.min if config.metric == "MSE" else np.max
    get_best_epoch = np.argmin if config.metric == "MSE" else np.argmax

    report = 'Best scores: \n Train set -> Loss: {:.4f}, {}: {:.4f}, Epoch: {} \n  Test set -> Loss: {:.4f}, {}: {:.4f}, Epoch: {} \n Config: \n{}'.format(
        np.min(epoch_train_losses), config.metric, get_best_score(epoch_train_scores), get_best_epoch(epoch_train_scores),
        np.min(epoch_test_losses), config.metric, get_best_score(epoch_test_scores), get_best_epoch(epoch_test_scores),
        config
    )

    with open(path.replace(".png", ".txt"), 'w') as file:
        file.write(report)
    
    