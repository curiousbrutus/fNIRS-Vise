#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages 

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn import model_selection
from sklearn import metrics
from yellowbrick.classifier import ClassificationReport
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from scipy import stats
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/home/jobbe/Mind-fMRI/data_anal/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# load data
data = pd.read_csv('/home/jobbe/Mind-fMRI/data_anal/kaggle/input/data_1_1.csv')


# In[3]:


# display data
data.head()


# In[4]:


# basic statistics
data.describe()


# In[5]:


sns.countplot(data=data, x='1')
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Count of Type')
plt.show()


# In[6]:


# Extract only HHb readings from the 'data' DataFrame
dataHonly = data.iloc[:, [1, 3, 5, 7, 9, 11, 13, 15]]
dataHonly.head()


# In[7]:


#extract O2 data 
dataO2only = data.iloc[:, 0:-1:2] #only O2Hb ratings
dataO2only.head()


# In[8]:


#attempt to get rid of outliers
def remove_outliers(df, z_threshold=1.5):
    # Calculate the absolute z-score for each column
    z_scores = stats.zscore(df, axis=0, nan_policy='omit')
    
    # Check if any of the z-scores exceed the threshold in any row
    is_outlier_row = (abs(z_scores) > z_threshold).any(axis=1)
    
    # Filter out rows that have any outlier in the columns
    df_cleaned = df[~is_outlier_row]
    
    return df_cleaned
  
# Remove rows containing any outlier in the HHb readings for the smaller dataset
dataHonly_cleaned = remove_outliers(dataHonly)
dataHonly_cleaned_type = pd.concat([dataHonly_cleaned,data.iloc[:, -1:]], axis=1)

dataHonly_cleaned_type.head()


# In[9]:


dataO2only_cleaned = remove_outliers(dataO2only)
dataO2only_cleaned_type = pd.concat([dataO2only_cleaned,data.iloc[:, -1:]], axis=1)

dataO2only_cleaned_type.head()


# Correlation Matrix

# In[10]:


# Calculate the correlation matrix for HHb readings
corr_n_backHonly =  dataHonly_cleaned.corr()

# Create a heatmap to visualize the correlation matrix
sns.heatmap(corr_n_backHonly, vmin=None, vmax=0.995, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("HHb Correlation Matrix Heatmap")
plt.show()


# In[11]:


corr_n_backO2only = dataO2only_cleaned.corr()
sns.heatmap(corr_n_backO2only, vmin=None,vmax=0.995, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("HO2 Correlation Matrix Heatmap")
plt.show()


# In[12]:


data = pd.concat([dataO2only_cleaned, dataHonly_cleaned], axis = 1)
data.describe()


# In[13]:


corr_n_back = data.corr()
sns.heatmap(corr_n_back, vmin=None, vmax=0.995, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Reorganized Correlation Matrix Heatmap")
plt.show()


# In[14]:


featuresO2 = dataO2only_cleaned_type.columns[:]
print(featuresO2)


# In[15]:


featuresH = dataHonly_cleaned_type.columns[:]
print(featuresH)


# In[16]:


data_small_O2 = dataO2only_cleaned_type.iloc[::75,:]
data_small_O2.head()


# In[17]:


data_small_H = dataHonly_cleaned_type.iloc[::75, :]
data_small_H.head()


# In[18]:


#commenting this cell out for timing reasons
#def swarm_plots(data_in):
    #featuresO2 = data_in.columns[:-1]  # Exclude the last column ('Type') (will create all of the swarm plots)
    #features = data_in.columns[0:2] # Use only the first two columns (will only create two swarm plots)
    
    #nback_colors = {'1-back': 'blue', '3-back': 'orange'}

    #for feature in featuresO2:
    #plt.figure()
       # sns.swarmplot(data=data_in, x='Type', y=feature, palette=nback_colors.values())
      #  plt.xlabel('Type')
       # plt.ylabel(feature)
        #plt.title(f'Swarm Plot of {feature}')
      #  plt.show()

# Call the function to generate swarm plots for each feature
#swarm_plots(data_small_O2)


# In[19]:


# Swarm plot deoxy
#swarm_plots(data_small_H)


# In[20]:


# Load data
df = pd.read_csv('/home/jobbe/Mind-fMRI/data_anal/kaggle/input/data_1_1.csv')

#remove outliers
data = remove_outliers(data)


# In[21]:


# Preprocessing
df['Type'] = pd.factorize(df["Type"])[0] # convert species to numerical

# Split data into features (X) and target (y)
X = df.drop(columns=['Type'])

# Label encode target
encoder = LabelEncoder()
y = encoder.fit_transform(df['Type'])


# In[22]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# Hyperparameter tuning
param_grid = {'solver': ['svd', 'lsqr', 'eigen'],
              'tol': [0.0001, 0.0002, 0.0003]}

lda = LinearDiscriminantAnalysis()

grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Parameters: ", grid_search.best_params_)


# In[24]:


# Fit the model with best parameters
lda = LinearDiscriminantAnalysis(solver=grid_search.best_params_['solver'], tol=grid_search.best_params_['tol'])
lda.fit(X_train, y_train)


# In[25]:


# Make predictions
y_pred = lda.predict(X_test)

# Model evaluation
print(classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=lda.classes_, yticklabels=lda.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[26]:


# Visualizing with Classification Report
visualizer = ClassificationReport(lda, support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# In[27]:


# Get predicted probabilities
y_probs = lda.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[28]:


# ROC curve
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
y_pred_binarized = label_binarize(y_pred, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[29]:


# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(lda, X, y, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy Score')
plt.legend(loc='best')
plt.show()

