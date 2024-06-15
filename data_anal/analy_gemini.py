import pandas as pd

# Assuming your Excel files are named "subject1.xlsx", "subject2.xlsx", etc.
subject_data = {}
for i in range(1, number_of_subjects+1):
  filename = f"subject{i}.xlsx"
  subject_data[f"subject{i}"] = pd.read_excel(filename)

# Assuming sampling rate of 10 Hz and 16-minute recording
sampling_rate = 10
recording_time_minutes = 16
total_datapoints = recording_time_minutes * 60 * sampling_rate

timebase = pd.to_timedelta(range(len(subject_data['subject1'])) / sampling_rate, unit='s')

# Add timebase as a new column in your pandas dataframe
subject_data['subject1']['Time'] = timebase

# Repeat for other subjects


# Assuming columns 2- onwards contain channel data
image_categories = ["COCO", "Imagenet", "Scene"]  # Modify based on your categories
block_averages = {}
for subject, data in subject_data.items():
  block_averages[subject] = {}
  for category in image_categories:
    # Identify block indices based on category timings (implement based on your experiment design)
    block_indices = # Implement logic to find block indices for this category
    block_average = data.iloc[block_indices, 1:].mean(axis=0)  # Assuming channels from col 2 onwards
    block_averages[subject][category] = block_average


# Assuming block_averages dictionary stores category-wise block averages
features = {}
for subject, category_data in block_averages.items():
  features[subject] = {}
  for category, block_average in category_data.items():
    # Extract features from block_average (e.g., mean, std)
    mean_feature = block_average.mean()
    std_feature = block_average.std()
    features[subject][category] = [mean_feature, std_feature]  # Modify based on features extracted



# Assuming segment_length is the number of data points per labeled segment
segment_length = # Define segment length based on your experiment design
sequence_data = {}
for subject, category_data in block_averages.items():
  sequence_data[subject] = {}
  for category, block_average in category_data.items():
    # Reshape block_average into segments
    segments = np.array([block_average[i:i+segment_length] for i in range(0, len(block_average), segment_length)])
    sequence_data[subject][category] = segments

# Plot the sequence data
for subject, category_data in sequence_data.items():
  for category, segments in category_data.items():
    plt.figure()
    plt.title(f"{subject} - {category}")
    for segment in segments:
      plt.plot(segment)
    plt.show()

