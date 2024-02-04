import os
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
from scipy.io import arff
import inspect
import sys
# Get the root directory of your project (the directory containing 'src' and 'plugins')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Set the device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Configure Polars
pl.Config.set_tbl_rows(-1)

print('Loading Data')
# Original Data
train, trameta = arff.loadarff('data/GCM_Training.arff')
test, tstmeta = arff.loadarff('data/GCM_Test.arff')
train = pl.from_numpy(train)
test = pl.from_numpy(test)
train_columns = train.columns
test_columns = test.columns
TRAIN = train.to_numpy()
TEST = test.to_numpy()

print('Scale original data')
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

#  Add more data to test-set
increase_test_set = False

if increase_test_set:
    #Extract observation from train and add to test
    num_samples = 20  # Example number
    sample_indices = np.random.choice(train_original.shape[0], size=num_samples, replace=False)
    sampled_data = train_original[sample_indices]
    test_original = np.concatenate((test_original, sampled_data), axis=0)
    train_original = np.delete(train_original, sample_indices, axis=0)

# load sythetic
X_syn = np.load(currentdir + "/synthetic_data_list.npy", allow_pickle=True)
y_syn = np.load(currentdir + "/labels_list.npy", allow_pickle=True)
# Reshape X_syn to 2D
X_syn_reshaped = X_syn.reshape(-1, X_syn.shape[2])  # Shape will be (14*16, 16063)
# Reshape y_syn to have the same number of rows as X_syn_reshaped
y_syn_reshaped = y_syn.reshape(-1, 1)  # Shape will be (14*16, 1)
# Join X_syn_reshaped and y_syn_reshaped
train_syn = np.concatenate((X_syn_reshaped, y_syn_reshaped), axis=1)  # Shape will be (14*16, 16064)
# Concatenate train_syn with train
combined_train = np.concatenate((train_original, train_syn), axis=0)
# Now combined_train has the combined data

# Convert the NumPy array to a Pandas DataFrame
train = pd.DataFrame(combined_train, columns=train_columns)
test = pd.DataFrame(test_original, columns=test_columns)

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
y_train, unique_classes = pd.factorize(y_train)
y_test = y_test.str.decode('utf-8')
y_test, unique_classes = pd.factorize(y_test)

# Define the pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    #("scaler", StandardScaler()),    
    ("SVC", SVC(class_weight='balanced', decision_function_shape='ovo')),
      
        # Adjust parameters as needed
])

print('Fiting')
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Use the pipeline to make predictions on the test set
y_pred = pipeline.predict(X_test)

# Generating classification report
report = classification_report(y_test, y_pred, target_names=unique_classes)

# Print evaluation report
print("Evaluation Report for MLP Classifier with StandardScaler:")
print(report)

weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Overall Weighted F1 Score: {weighted_f1}")

withgraph = False

if withgraph:   
    report = classification_report(y_test, y_pred, target_names=unique_classes, output_dict=True)
    f1_scores = {label: report[label]['f1-score'] for label in unique_classes}
    categories = list(f1_scores.keys())
    scores = list(f1_scores.values())
    plt.figure(figsize=(12, 6))
    plt.bar(categories, scores)
    plt.xlabel('Categories')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.title('F1 Scores for Each Category')
    plt.show()

# With 66 Test-Observation
"""
precision    recall  f1-score   support

       Breast       0.17      0.20      0.18         5
     Prostate       0.00      0.00      0.00         3
         Lung       0.00      0.00      0.00         4
   Colorectal       0.00      0.00      0.00         5
     Lymphoma       0.00      0.00      0.00        10
      Bladder       0.25      0.25      0.25         4
     Melanoma       0.00      0.00      0.00         3
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       0.19      1.00      0.32        10
        Renal       0.00      0.00      0.00         3
     Pancreas       0.00      0.00      0.00         3
        Ovary       0.00      0.00      0.00         3
 Mesothelioma       1.00      0.67      0.80         3
          CNS       0.00      0.00      0.00         8

     accuracy                           0.21        66
    macro avg       0.11      0.15      0.11        66
 weighted avg       0.10      0.21      0.11        66

Overall Weighted F1 Score: 0.11416511152581535
"""

# With 46 Test-Observation
"""
precision    recall  f1-score   support

       Breast       0.17      0.20      0.18         5
     Prostate       0.00      0.00      0.00         3
         Lung       0.00      0.00      0.00         4
   Colorectal       0.00      0.00      0.00         5
     Lymphoma       0.00      0.00      0.00        10
      Bladder       0.25      0.25      0.25         4
     Melanoma       0.00      0.00      0.00         3
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       0.19      1.00      0.32        10
        Renal       0.00      0.00      0.00         3
     Pancreas       0.00      0.00      0.00         3
        Ovary       0.00      0.00      0.00         3
 Mesothelioma       1.00      0.67      0.80         3
          CNS       0.00      0.00      0.00         8

     accuracy                           0.21        66
    macro avg       0.11      0.15      0.11        66
 weighted avg       0.10      0.21      0.11        66

Overall Weighted F1 Score: 0.11416511152581535
"""