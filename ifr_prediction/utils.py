from box import Box
from matplotlib import pyplot as plt
import numpy as np
import yaml
import os

def get_config(file_path):
    """Load a YAML file and return a Box object (dot notation)"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return Box(data)

def plot_performance(model_name, config, epoch_train_losses, epoch_test_losses, epoch_train_scores, epoch_test_scores):
    fig = plt.figure(figsize=(10, 4))

    # plot losses
    plt.subplot(121)
    plt.plot(np.arange(1, config.epochs + 1), np.array(epoch_train_losses)[:, -1])  # train loss (on epoch end)
    plt.plot(np.arange(1, config.epochs + 1), epoch_test_losses)         #  test loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc="upper left")

    # plot mse
    plt.subplot(122)
    plt.plot(np.arange(1, config.epochs + 1), np.array(epoch_train_scores)[:, -1])  # train accuracy (on epoch end)
    plt.plot(np.arange(1, config.epochs + 1), epoch_test_scores)         #  test accuracy (on epoch end)
    plt.title("training scores")
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend(['train', 'test'], loc="upper left")

    path = os.path.join(config.charts_dir, f"{model_name}.png")
    plt.savefig(path, dpi=600)
    plt.close(fig)
    # plt.show()