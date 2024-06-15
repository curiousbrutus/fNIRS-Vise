# fNIRS data analysis with MNE-Python

# load necessery libraries for mne-nirs

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Load data from text file
#data = np.genfromtxt('/home/jobbe/Mind-fMRI/data_anal/eyyub_2_file.txt', missing_values="NULL", filling_values=np.nan)
#import pandas as pd

# Read the data from the file
data = pd.read_csv('/home/jobbe/Mind-fMRI/data_anal/eyyub_2_file.txt', delimiter='\t')

# Perform basic analysis
print("Number of rows:", len(data))
print("Column names:", data.columns)
print("Summary statistics:")
print(data.describe())


'''
# clean the data
data = data[~np.isnan(data).any(axis=1)]

# prepare the data for MNE , channel types should be 'fnirs_cw_amplitude' or 'fnirs_od' for fNIRS data
data = data.T
ch_names = ['CH' + str(i+1) for i in range(data.shape[0])]
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='fnirs_cw_amplitude')
raw = mne.io.RawArray(data, info)

# plot the data
raw.plot()
'''