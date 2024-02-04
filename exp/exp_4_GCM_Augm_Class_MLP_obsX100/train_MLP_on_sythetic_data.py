import os
import json
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.neural_network import MLPClassifier
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
    num_samples = 50  # Example number
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

print(f'train shape: {train.shape}')
print(f'test shape: {test.shape}')

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

max_iter = 500
#max_iter = 1000
#max_iter = 2000

pipeline = Pipeline([
    #("scaler", StandardScaler()),    
    ("mlp", MLPClassifier(max_iter=max_iter, 
                          random_state=1, 
                          hidden_layer_sizes=[100, 100],
                          activation='relu'
                          ))  # Adjust parameters as needed
    #("SVC", SVC(class_weight='balanced', decision_function_shape='ovo')),
      
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

# With 46 Test-Observation and 500 epoch
"""
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.00      0.00      0.00         3
     Prostate       0.00      0.00      0.00         2
         Lung       1.00      0.67      0.80         3
   Colorectal       0.00      0.00      0.00         3
     Lymphoma       1.00      0.50      0.67         6
      Bladder       0.10      0.33      0.15         3
     Melanoma       1.00      0.50      0.67         2
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       1.00      0.83      0.91         6
        Renal       0.67      0.67      0.67         3
     Pancreas       0.50      0.33      0.40         3
        Ovary       1.00      0.33      0.50         3
 Mesothelioma       1.00      1.00      1.00         3
          CNS       0.80      1.00      0.89         4

     accuracy                           0.50        46
    macro avg       0.58      0.44      0.48        46
 weighted avg       0.65      0.50      0.54        46

Overall Weighted F1 Score: 0.5414124522820175
"""

# With 46 Test-Observation and 1000 epoch
"""
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.00      0.00      0.00         3
     Prostate       0.00      0.00      0.00         2
         Lung       1.00      0.67      0.80         3
   Colorectal       0.00      0.00      0.00         3
     Lymphoma       1.00      0.50      0.67         6
      Bladder       0.10      0.33      0.15         3
     Melanoma       1.00      0.50      0.67         2
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       1.00      0.83      0.91         6
        Renal       0.67      0.67      0.67         3
     Pancreas       0.50      0.33      0.40         3
        Ovary       1.00      0.33      0.50         3
 Mesothelioma       1.00      1.00      1.00         3
          CNS       0.80      1.00      0.89         4

     accuracy                           0.50        46
    macro avg       0.58      0.44      0.48        46
 weighted avg       0.65      0.50      0.54        46

Overall Weighted F1 Score: 0.5414124522820175
"""

# 46 Test Observation and 2000 EPOCH
"""
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.00      0.00      0.00         3
     Prostate       0.00      0.00      0.00         2
         Lung       1.00      0.67      0.80         3
   Colorectal       0.00      0.00      0.00         3
     Lymphoma       1.00      0.50      0.67         6
      Bladder       0.10      0.33      0.15         3
     Melanoma       1.00      0.50      0.67         2
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       1.00      0.83      0.91         6
        Renal       0.67      0.67      0.67         3
     Pancreas       0.50      0.33      0.40         3
        Ovary       1.00      0.33      0.50         3
 Mesothelioma       1.00      1.00      1.00         3
          CNS       0.80      1.00      0.89         4

     accuracy                           0.50        46
    macro avg       0.58      0.44      0.48        46
 weighted avg       0.65      0.50      0.54        46

Overall Weighted F1 Score: 0.5414124522820175
"""

# 46 Test Observation and 500 EPOCH, bigger net
#  hidden_layer_sizes=[100, 100], activation='relu'

"""
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.00      0.00      0.00         3
     Prostate       0.00      0.00      0.00         2
         Lung       1.00      0.33      0.50         3
   Colorectal       0.50      1.00      0.67         3
     Lymphoma       1.00      0.83      0.91         6
      Bladder       0.22      0.67      0.33         3
     Melanoma       1.00      1.00      1.00         2
Uterus__Adeno       0.00      0.00      0.00         2
     Leukemia       1.00      0.83      0.91         6
        Renal       0.00      0.00      0.00         3
     Pancreas       0.67      0.67      0.67         3
        Ovary       0.67      0.67      0.67         3
 Mesothelioma       1.00      0.67      0.80         3
          CNS       1.00      1.00      1.00         4

     accuracy                           0.61        46
    macro avg       0.58      0.55      0.53        46
 weighted avg       0.66      0.61      0.60        46

Overall Weighted F1 Score: 0.6045454545454545
"""





# More test sample experiment---------------------------

# With 66 Test-Observation
"""
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.17      0.25      0.20         4
     Prostate       1.00      0.40      0.57         5
         Lung       0.25      0.33      0.29         3
   Colorectal       0.44      1.00      0.62         4
     Lymphoma       1.00      0.71      0.83         7
      Bladder       0.00      0.00      0.00         4
     Melanoma       0.50      0.33      0.40         3
Uterus__Adeno       0.20      0.25      0.22         4
     Leukemia       1.00      0.89      0.94         9
        Renal       0.00      0.00      0.00         4
     Pancreas       0.25      0.75      0.38         4
        Ovary       0.00      0.00      0.00         5
 Mesothelioma       0.80      1.00      0.89         4
          CNS       1.00      0.67      0.80         6

     accuracy                           0.52        66
    macro avg       0.47      0.47      0.44        66
 weighted avg       0.56      0.52      0.50        66

Overall Weighted F1 Score: 0.50339682104388
"""

# With 96 Test Observation and 500 epoch

"""
train shape: (1494, 16064)
test shape: (96, 16064)
Evaluation Report for MLP Classifier with StandardScaler:
               precision    recall  f1-score   support

       Breast       0.00      0.00      0.00         3
     Prostate       0.50      0.60      0.55         5
         Lung       0.00      0.00      0.00         6
   Colorectal       1.00      0.12      0.22         8
     Lymphoma       0.91      0.77      0.83        13
      Bladder       0.11      0.17      0.13         6
     Melanoma       0.44      0.80      0.57         5
Uterus__Adeno       0.50      0.33      0.40         6
     Leukemia       1.00      0.81      0.90        16
        Renal       0.40      0.80      0.53         5
     Pancreas       0.33      0.80      0.47         5
        Ovary       0.50      0.33      0.40         6
 Mesothelioma       1.00      0.75      0.86         4
          CNS       1.00      0.88      0.93         8

     accuracy                           0.56        96
    macro avg       0.55      0.51      0.49        96
 weighted avg       0.65      0.56      0.56        96

Overall Weighted F1 Score: 0.5630750022928014
"""
