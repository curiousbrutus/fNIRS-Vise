# Data labeling based on event timings for fNIRS data in python using pandas 

import pandas as pd

# Load your fNIRS data from the Excel file
data = pd.read_excel('/home/jobbe/Mind-fMRI/data_anal/data_trial.xlsx')

# Define event timing based on your experiment description
event_timings = {
    'COCO': [0, 2250],
    'blank_screen1': [2250, 2300],
    'ImageNet': [2340, 4580],
    'blank_screen2': [4580, 4620],
    'Scene': [4650, 6850],
}

def label_data(data, event_timings):
    for event, timing in event_timings.items():
        if timestamp >= timing[0] and timestamp < timing[1]:
            event_label = event
            break
        labels.append(event_label)
    return labels

# Add event labels to your data
data['event'] = label_data(data, event_timings)

# Now you have labeled data segments, and you can proceed with further analysis
print(data.head())
