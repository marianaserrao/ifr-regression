import os, cv2
from tqdm import tqdm
import numpy as np

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
    
def mean_pixel_value(image):
    # Compute the mean value of the pixels in the image
    flattened_array = image.flatten()

    # Calculate the mean of the pixel values
    mean_value = np.mean(flattened_array)

    # Calculate the variance
    variance = np.var(flattened_array)
    return variance
    
def variance_of_laplacian(image):
    # Compute the Laplacian of the image and then return the variance
    return cv2.Laplacian(image, cv2.CV_64F).var()
    
def get_image_clarity_and_mean_list(folder_path):
    files = list_files_in_folder(folder_path)
    image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'))]
    image_files = sorted(image_files)
    
    clarity_mean_list = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image_id = image_file.split("_")[3]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        clarity = variance_of_laplacian(image)
        mean_value = mean_pixel_value(image)
        clarity_mean_list.append((clarity, mean_value, image_file))
    
    # Sort the list by clarity score in descending order
    clarity_mean_list.sort(reverse=True, key=lambda x: x[1])
    
    return clarity_mean_list

# Example usage
folder_path = '/media/jlsstorage/masstorage/angiograms/Videos/780/8'
# files = list_files_in_folder(folder_path)
clarity_mean_list = get_image_clarity_and_mean_list(folder_path)
for clarity, mean_value, image_name in clarity_mean_list:
    print(f"{image_name}: Clarity = {clarity}, Mean Pixel Value = {mean_value}")