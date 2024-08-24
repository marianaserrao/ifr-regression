from train import *
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    train_crnn()