# This is the python file for image segmentation and preproccesing in fNIRS data

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import scipy.io as sio
import scipy.ndimage as ndimage
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import binary_dilation

def preproccesing(data, fs, lowcut, highcut, order, threshold, min_size, max_size, dilation_size, save_path):
    """
    Preproccesing data for image segmentation
    :param data: fNIRS data
    :param fs: sampling frequency
    :param lowcut: lowcut frequency
    :param highcut: highcut frequency
    :param order: order of the filter
    :param threshold: threshold for the binary image
    :param min_size: minimum size of the region
    :param max_size: maximum size of the region
    :param dilation_size: size of the dilation
    :param save_path: path to save the preproccesed data
    :return: preproccesed data
    """
    # Filter data
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    # Normalize data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # Create binary image
    binary_image = data > threshold
    # Remove small regions
    binary_image = remove_small_regions(binary_image, min_size, max_size)
    # Dilation
    binary_image = binary_dilation(binary_image, structure=np.ones((dilation_size, dilation_size)))
    # Save data
    sio.savemat(save_path, {'data': binary_image})
    return binary_image

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butterworth bandpass filter
    :param lowcut: lowcut frequency
    :param highcut: highcut frequency
    :param fs: sampling frequency
    :param order: order of the filter
    :return: b, a
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply Butterworth bandpass filter
    :param data: fNIRS data
    :param lowcut: lowcut frequency
    :param highcut: highcut frequency
    :param fs: sampling frequency
    :param order: order of the filter
    :return: filtered data
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def remove_small_regions(binary_image, min_size, max_size):
    """
    Remove small regions from the binary image
    :param binary_image: binary image
    :param min_size: minimum size of the region
    :param max_size: maximum size of the region
    :return: binary image
    """
    labeled_array, num_features = label(binary_image)
    sizes = ndimage.sum(binary_image, labeled_array, range(num_features + 1))
    mask_size = sizes < min_size
    remove_pixel = mask_size[labeled_array]
    binary_image[remove_pixel] = 0
    mask_size = sizes > max_size
    remove_pixel = mask_size[labeled_array]
    binary_image[remove_pixel] = 0
    return binary_image

def plot_data(data, fs, lowcut, highcut, order, threshold, min_size, max_size, dilation_size, save_path):
    """
    Plot the preproccesed data
    :param data: fNIRS data
    :param fs: sampling frequency
    :param lowcut: lowcut frequency
    :param highcut: highcut frequency
    :param order: order of the filter
    :param threshold: threshold for the binary image
    :param min_size: minimum size of the region
    :param max_size: maximum size of the region
    :param dilation_size: size of the dilation
    :param save_path: path to save the preproccesed data
    """
    # Filter data
    data = butter_bandpass_filter(data, lowcut, highcut, fs, order)
    # Normalize data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # Create binary image
    binary_image = data > threshold
    # Remove small regions
    binary_image = remove_small_regions(binary_image, min_size, max_size)
    # Dilation
    binary_image = binary_dilation(binary_image, structure=np.ones((dilation_size, dilation_size)))
    # Plot data
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(data)
    ax[0].set_title('fNIRS data')
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].set_title('Binary image')
    plt.savefig(save_path)
    plt.show()

def main():
    # Load data
    data = np.load('data.npy')
    # Parameters
    fs = 10
    lowcut = 0.01
    highcut = 0.1
    order = 5
    threshold = 0.5
    min_size = 10
    max_size = 100
    dilation_size = 5
    save_path = 'preproccesed_data.mat'
    # Preproccesing
    preproccesing(data, fs, lowcut, highcut, order, threshold, min_size, max_size, dilation_size, save_path)
    # Plot data
    plot_data(data, fs, lowcut, highcut, order, threshold, min_size, max_size, dilation_size, 'preproccesed_data.png')

if __name__ == '__main__':
    main()