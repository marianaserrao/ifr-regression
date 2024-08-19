from vessel_segmentation import VesselSegmentation

import os
from tqdm import tqdm
import numpy as np
import re

def natural_sort_key(s):
    # This splits the string into a list of strings and integers
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def list_files_in_folder(folder_path):
    try:
        # List all files and directories in the specified folder
        files_and_dirs = os.listdir(folder_path)
        
        # Filter out directories, keeping only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
        # files = [f for f in files_and_dirs if os.path.isdir(os.path.join(folder_path, f))]
        
        return files
    except Exception as e:
        return str(e)
    
vs = VesselSegmentation()

folder="/media/jlsstorage/masstorage/angiograms/Videos/780/5"
output_dir = "vessel_segmentation/output"
vessel_sizes = []

files = list_files_in_folder(folder)
files = sorted(files, key=natural_sort_key)
for i,file in tqdm(enumerate(files)):
    file_path = os.path.join(folder, file)
    image = vs.read_image(file_path)
    mask = vs.predict_mask(image)
    vessel_size = np.count_nonzero(mask == 2)
    vessel_sizes.append(vessel_size)
    # vs.save_mask(mask, os.path.join(output_dir, file))

with open('vessels.txt', 'w') as file:
    for item in vessel_sizes:
        file.write(f"{item}\n")
