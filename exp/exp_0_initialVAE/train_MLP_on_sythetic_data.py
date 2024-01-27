import os
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
from scipy.io import arff

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure Polars
pl.Config.set_tbl_rows(-1)

# Original Data
train, trameta = arff.loadarff('/home/sebacastillo/ealab/data/GCM_Training.arff')
test, tstmeta = arff.loadarff('/home/sebacastillo/ealab/data/GCM_Test.arff')
train = pl.from_numpy(train)
test = pl.from_numpy(test)
train_columns = train.columns
test_columns = test.columns
TRAIN = train.to_numpy()
TEST = test.to_numpy()

#Scale
scaler = StandardScaler()
X_TRAIN = TRAIN[:,:-1]
y_train = TRAIN[:,-1]
#y_train = np.where(np.array(y_train) == 2, 1, 0).astype('int64')
X_TEST = TEST[:,:-1]
y_test = TEST[:,-1]
#y_test = np.where(np.array(y_test) == 2, 1, 0).astype('int64')
scaler.fit(X_TRAIN)
scale_train = scaler.transform(X_TRAIN)
scale_test = scaler.transform(X_TEST)

# Recombine original dataa
train_original = np.concatenate((scale_train, y_train.reshape(-1,1)), axis=1)  
test_original = np.concatenate((scale_test, y_test.reshape(-1,1)), axis=1)  

# load sythetic
X_syn = np.load("/home/sebacastillo/ealab/exp/exp_0_initialVAE/synthetic_data_list.npy", allow_pickle=True)
y_syn = np.load("/home/sebacastillo/ealab/exp/exp_0_initialVAE/labels_list.npy", allow_pickle=True)
# Reshape X_syn to 2D
X_syn_reshaped = X_syn.reshape(-1, X_syn.shape[2])  # Shape will be (14*16, 16063)
# Reshape y_syn to have the same number of rows as X_syn_reshaped
y_syn_reshaped = y_syn.reshape(-1, 1)  # Shape will be (14*16, 1)
# Join X_syn_reshaped and y_syn_reshaped
train_syn = np.concatenate((X_syn_reshaped, y_syn_reshaped), axis=1)  # Shape will be (14*16, 16064)

# Concatenate train_syn with train
combined_train = np.concatenate((train_original, train_syn), axis=0)
# Now combined_train has the combined data

# Decide the number of samples you want to extract
num_samples = 40  # Example number
# Sample without replacement
sample_indices = np.random.choice(combined_train.shape[0], size=num_samples, replace=False)

# Extract the samples for augmentation
sampled_data = combined_train[sample_indices]
combined_test = np.concatenate((test_original, sampled_data), axis=0)

# Remove the sampled observations from combined_train
combined_train = np.delete(combined_train, sample_indices, axis=0)


# Convert the NumPy array to a Pandas DataFrame
train = pd.DataFrame(combined_train, columns=train_columns)
test = pd.DataFrame(combined_test, columns=test_columns)


# Split
from sklearn.model_selection import train_test_split
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
# Splitting the 'test' DataFrame into features and target
# Also assuming the last column is the target here
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


# Convert byte strings to regular strings (if necessary)
y_train = y_train.str.decode('utf-8')
# Factorize the class column to map each unique string to a unique integer
y_train, unique_classes = pd.factorize(y_train)
# Now _train contains integer representations of your classes
# 'unique_classes' holds the unique class labels in order
print(y_train)  # This will show the integer-encoded classes
print(unique_classes)  

# Convert byte strings to regular strings (if necessary)
y_test = y_test.str.decode('utf-8')
# Factorize the class column to map each unique string to a unique integer
y_test, unique_classes = pd.factorize(y_test)
# Now _test contains integer representations of your classes
# 'unique_classes' holds the unique class labels in order
print(y_test)  # This will show the integer-encoded classes
print(unique_classes)  

# One-hot encode the labels
# mlb = MultiLabelBinarizer()
# y_train_encoded = mlb.fit_transform(y_train)
# y_test_encoded = mlb.transform(y_test)


# Define the pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=500, random_state=1))  # Adjust parameters as needed
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Use the pipeline to make predictions on the test set
y_pred = pipeline.predict(X_test)

# Generating classification report
report = classification_report(y_test, y_pred, target_names=unique_classes)

# Print evaluation report
print("Evaluation Report for MLP Classifier with StandardScaler:")
print(report)

