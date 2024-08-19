#mean = 110
#std = 37.8
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_mean_and_std(json_file):
    sum_channels = np.zeros(3)
    sum_squared_diff_channels = np.zeros(3)
    pixel_count = 0

    # Load and parse JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # First pass: Compute the mean
    for entry in tqdm(data["kf_exams"]):
        directory = entry.get('path', None)
        if directory and os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.png'):
                    img_path = os.path.join(directory, filename)
                    img = Image.open(img_path)
                    img_np = np.array(img)

                    if img_np.ndim == 3 and img_np.shape[2] == 3:
                        sum_channels += np.sum(img_np, axis=(0, 1))
                        pixel_count += img_np.shape[0] * img_np.shape[1]
        else:
            print(f"Directory {directory} not found or invalid.")

    mean_channels = sum_channels / pixel_count

    # Second pass: Compute the sum of squared differences from the mean
    for entry in tqdm(data["kf_exams"]):
        directory = entry.get('path', None)
        if directory and os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.png'):
                    img_path = os.path.join(directory, filename)
                    img = Image.open(img_path)
                    img_np = np.array(img)

                    if img_np.ndim == 3 and img_np.shape[2] == 3:
                        sum_squared_diff_channels += np.sum((img_np - mean_channels) ** 2, axis=(0, 1))

    # Compute standard deviation
    std_channels = np.sqrt(sum_squared_diff_channels / pixel_count)

    print(f"Overall mean channel values: R={mean_channels[0]}, G={mean_channels[1]}, B={mean_channels[2]}")
    print(f"Overall std channel values: R={std_channels[0]}, G={std_channels[1]}, B={std_channels[2]}")

# Usage example
json_file_path = '../clinical_data.json'
compute_mean_and_std(json_file_path)
