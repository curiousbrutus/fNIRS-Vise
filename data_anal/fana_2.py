import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load your fNIRS data from the Excel file
data = pd.read_excel('/home/jobbe/Mind-fMRI/data_anal/data_trial.xlsx')

# Extract features from the data (replace with your feature extraction method)
# For example, let's say we extract mean and standard deviation of each channel
features = data.groupby('event').agg(['mean', 'std'])

# Label your data based on the experimental conditions or events
# For demonstration purposes, let's assume we have 3 events: COCO, ImageNet, Scene
labels = {
    'COCO': 0,
    'ImageNet': 1,
    'Scene': 2
}

# Map event labels to data
data['label'] = data['event'].map(labels)

# Split data into training and testing sets
# Define the new_data variable and assign it the appropriate value
new_data = pd.read_excel('data_anal/new_data_trial.xlsx')

# Extract features from the new data (replace with your feature extraction method)
new_data_features = new_data.groupby('event').agg(['mean', 'std'])

# Continue with the rest of the code
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
new_predictions = clf.predict(new_data_features)
print(new_predictions)
