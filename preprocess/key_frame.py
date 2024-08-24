import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, label, binary_dilation, binary_erosion
from scipy.signal import find_peaks
import os
from tqdm import tqdm
import re

def find_largest_connected_component(array, dilation_size=10):
    """
    Find the largest connected component of an array,
    considering custom connectivity defined by dilating the mask.

    Parameters:
    - array (np.ndarray): The input array.
    - dilation_size (int): The size of the dilation to apply for connecting components.

    Returns:
    - final_mask (np.ndarray): A binary mask of the largest connected component.
    - component_size (int): The size of the largest connected component.
    """

    # Dilate the binary mask to connect regions within the dilation size
    structuring_element = np.ones((dilation_size, dilation_size), dtype=int)
    dilated_mask = binary_dilation(array, structure=structuring_element)

    # Label connected components in the dilated mask
    sequences_mask, num_sequences = label(dilated_mask)

    # Measure sizes of connected components
    component_sizes = np.bincount(sequences_mask.ravel())
    component_sizes[0] = 0  # Set the background size to 0

    # Find the largest component
    largest_component_label = component_sizes.argmax()

    # Create a mask for the largest connected component
    largest_component_mask = (sequences_mask == largest_component_label) if largest_component_label!=0 else sequences_mask
    final_mask = binary_erosion(largest_component_mask, structure=structuring_element)

    max_component_size = np.sum(final_mask)

    return final_mask, max_component_size

def smooth_sequence(sequence, sigma=2):
    """
    Apply Gaussian smoothing to a sequence.
    
    Parameters:
    - sequence: list or numpy array of numbers
    - sigma: standard deviation for Gaussian kernel, controls the smoothing level (default is 5)
    
    Returns:
    - Smoothed sequence as a numpy array
    """

    smoothed_sequence = gaussian_filter1d(sequence, sigma=sigma)

    # cumsum = np.cumsum(np.insert(sequence, 0, 0)) 
    # smoothed_sequence = (cumsum[sigma:] - cumsum[:-sigma]) / sigma
    
    # # To handle the edges where the window does not fit
    # smoothed_sequence = np.concatenate((
    #     np.full(sigma-1, 0),  # Fill edges with NaN or some other value
    #     smoothed_sequence
    # ))
    return smoothed_sequence

def get_sequence_slope(sequence):
    """
    Compute the slope using the backward difference method.
    
    Parameters:
    - sequence: list or numpy array of numbers
    
    Returns:
    - slope: the slope (angular coefficients) of the sequence using the backward difference method
    """
    slope = np.diff(sequence)
    slope = np.append(slope, slope[-1])  # Extend the slope array to match the length of the input
    return slope

def save_sequences_plot(original_sequence, smoothed_sequence, slope, slope_threshold=200, file_path='sequences_plot.png'):
    """
    Save the plot of the original sequence, the smoothed sequence, and the angular coefficients (slope) to a file.
    
    Parameters:
    - original_sequence: list or numpy array of numbers
    - smoothed_sequence: list or numpy array of numbers (output from gaussian_smoothing)
    - slope: list or numpy array of angular coefficients (slope)
    - file_path: name of the file where the plot will be saved (default is 'sequences_plot.png')
    """
    plt.figure(figsize=(14, 10))

    # Plot the original sequence
    plt.subplot(3, 1, 1)
    plt.plot(original_sequence, label='Original Sequence')
    plt.title('Original Sequence')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # Plot the smoothed sequence
    plt.subplot(3, 1, 2)
    plt.plot(smoothed_sequence, label='Smoothed Sequence', color='orange')
    plt.title('Smoothed Sequence')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)

    # Plot the angular coefficients (slope)
    plt.subplot(3, 1, 3)
    plt.plot(slope, label='Angular Coefficients', color='green')
    plt.axhline(y=slope_threshold, color='red', linestyle='--', label='Slope Threshold')
    plt.title('Angular Coefficients (Slope)')
    plt.xlabel('Index')
    plt.ylabel('Slope')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def select_keyframe(raw_values, smoothed_values, slope_values, slope_threshold=200, slope_window=(10, 5)):
    # Step 1: Find the index of the maximum value in the smoothed array
    max_smooth_idx = np.argmax(smoothed_values)
    
    # Step 2: Attempt to find the index of the closest slope value above the threshold that comes before the max smoothed value
    slope_candidates = np.where(slope_values[:max_smooth_idx] >= slope_threshold)[0]
    
    # Step 3: If a slope > threshold is found, use its index; otherwise, use max_smooth_idx
    if len(slope_candidates) > 0:
        slope_idx = slope_candidates[-1]
    else:
        slope_idx = max_smooth_idx
    
    # Step 4: Define the window around the slope index
    window_start = max(0, slope_idx - slope_window[0])
    window_end = min(len(raw_values), slope_idx + slope_window[1])
    
    # Step 5: Find the index of the maximum raw value within this window
    final_idx = np.argmax(raw_values[window_start:window_end]) + window_start

    # print(max_smooth_idx, slope_idx, final_idx)
    return final_idx

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

def get_exam_keyframe(exam_path, kf_mask_dir, vessel_segmentation, min_mask_size=3000):
    vessel_sizes = []
    masks = []
    frames = list_files_in_folder(exam_path)
    frames = sorted(frames, key=natural_sort_key)
    for i,frame in enumerate(frames):
        frame_path = os.path.join(exam_path, frame)
        image = vessel_segmentation.read_image(frame_path)
        mask = vessel_segmentation.predict_mask(image)
        connected_mask, size = find_largest_connected_component(mask, dilation_size=20)
        vessel_sizes.append(size)
        masks.append(mask)

    smoothed_vessel_sizes = smooth_sequence(vessel_sizes)
    sizes_slope = get_sequence_slope(smoothed_vessel_sizes)
    kf_index = select_keyframe(vessel_sizes, smoothed_vessel_sizes, sizes_slope)
    
    if vessel_sizes[kf_index]<min_mask_size:
        return None
    
    kf_mask = masks[kf_index]
    kf_path = os.path.join(exam_path, frames[kf_index])
    kf_mask_path = os.path.join(kf_mask_dir, frames[kf_index])

    # with open("./vessels.txt", "w") as file:
    #     for size in vessel_sizes:
    #         file.write(f"{size}\n")
    # save_sequences_plot(vessel_sizes,smoothed_vessel_sizes,sizes_slope)

    vessel_segmentation.save_mask(kf_mask, kf_mask_path)

    return kf_index, kf_path, kf_mask_path, kf_mask

