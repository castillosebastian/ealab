# This script is adapted from
# Frank Ceballos
# https://towardsdatascience.com/model-design-and-selection-with-scikit-learn-18a29041d02a

# Because my main objective is to process this data in the cloud 
# I have refactored some objects and added functions for ploting and saving results. I
# also add error handler in order to don't stop the excecution.

###############################################################################
#                          1. Importing Libraries                             #
###############################################################################
# For reading, visualizing, and preprocessing data
import os
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time  # Import the time module

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# Method to find the root directory (assuming .git is in the root)
def find_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # To avoid infinite loop
        if ".git" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None  # Or raise an error if the root is not found

root_dir = find_root_dir()

train_file_path = os.path.join(root_dir, 'data', 'gisette_train.csv')
test_file_path = os.path.join(root_dir, 'data', 'gisette_test.csv')
current_dir =  root_dir + '/exp/gisette_base_allfeatures/'

###############################################################################
#                                  Get data                                 #
###############################################################################

train = pl.read_csv(train_file_path, has_header = True)
test = pl.read_csv(test_file_path, has_header = True)

y_train = train.select(pl.col('class')).to_pandas()
X_train = train.select(pl.col('*').exclude('class')).to_pandas()

y_test = test.select(pl.col('class')).to_pandas()
X_test = test.select(pl.col('*').exclude('class')).to_pandas()

print('Get data completed')

###############################################################################
#                                Classifiers                                #
###############################################################################
# Create list of tuples with classifier label and classifier object
classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging": BaggingClassifier(),
    "Extra Trees Ensemble": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Ridge": RidgeClassifier(),
    "SGD": SGDClassifier(),
    "BNB": BernoulliNB(),
    "GNB": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(),
    "LSVC": LinearSVC(),
    "NuSVC": NuSVC(),
    "SVC": SVC(),
    "DTC": DecisionTreeClassifier(),
    "ETC": ExtraTreeClassifier()
}

# Create dict of decision function labels
DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}

# Create dict for classifiers with feature_importances_ attribute
FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}

print('Classifiers completed')
###############################################################################
#                             Hyper-parameters                             #
###############################################################################
parameters = {
    "LDA": {
        "classifier__solver": ["svd"]
    },
    "QDA": {
        "classifier__reg_param": [0.01 * ii for ii in range(0, 101)]
    },
    "AdaBoost": {
        "classifier__estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "classifier__n_estimators": [200],
        "classifier__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 1.0]
    },
    "Bagging": {
        "classifier__estimator": [DecisionTreeClassifier(max_depth=ii) for ii in range(1, 6)],
        "classifier__n_estimators": [200],
        "classifier__max_features": [0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        "classifier__n_jobs": [-1]
    },
    # Update dict with Gradient Boosting
    "Gradient Boosting": { 
                            "classifier__learning_rate":[0.15,0.1,0.01,0.001], 
                            "classifier__n_estimators": [200],
                            "classifier__max_depth": [2,3,4,5,6],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                            "classifier__max_features": [ "sqrt", "log2"],
                            "classifier__subsample": [0.8, 0.9, 1]
    },
    # Update dict with Extra Trees
    "Extra Trees Ensemble": { 
                            "classifier__n_estimators": [200],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": [ "sqrt", "log2"],
                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__n_jobs": [-1]
    },
    # Update dict with Random Forest Parameters
    "Random Forest": { 
                        "classifier__n_estimators": [200],
                        "classifier__class_weight": [None, "balanced"],
                        "classifier__max_features": [ "sqrt", "log2"],
                        "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                        "classifier__criterion" :["gini", "entropy"]     ,
                        "classifier__n_jobs": [-1]
    },
    # Update dict with Ridge
    "Ridge": { 
                "classifier__alpha": [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    },
    # Update dict with SGD Classifier
    "SGD": { 
            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__n_jobs": [-1]
    },
    # Update dict with BernoulliNB Classifier
    "BNB": { 
            "classifier__alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0]
    },
    # Update dict with GaussianNB Classifier
    "GNB": { 
            "classifier__var_smoothing": [1e-9, 1e-8,1e-7, 1e-6, 1e-5]
    },
    # Update dict with K Nearest Neighbors Classifier
    "KNN": { 
            "classifier__n_neighbors": list(range(1,31)),
            "classifier__p": [1, 2, 3, 4, 5],
            "classifier__leaf_size": [5, 10, 20, 30, 40, 50],
            "classifier__n_jobs": [-1]
    },
    # Update dict with MLPClassifier
    "MLP": { 
            "classifier__hidden_layer_sizes": [(5,5), (10,10), (5,5,5), (10,10,10)],
            "classifier__activation": ["identity", "logistic", "tanh", "relu"],
            "classifier__learning_rate": ["constant", "invscaling", "adaptive"],
            "classifier__max_iter": [500, 1000, 2000],
            "classifier__alpha": list(10.0 ** -np.arange(1, 10)),
    },
    "LSVC": { 
            "classifier__penalty": ["l2"],
            "classifier__C": [0.0001, 0.01, 0.1, 1.0, 10, 100]
    },
    "NuSVC": { 
            "classifier__nu": [0.25, 0.50, 0.75],
            "classifier__kernel": ["linear", "rbf", "poly"],
            "classifier__degree": [1,3,5,6],
    },
    "SVC": { 
            "classifier__kernel": ["linear", "rbf", "poly"],
            "classifier__gamma": ["auto"],
            "classifier__C": [0.1, 0.5, 1, 10, 50, 100],
            "classifier__degree": [1, 3, 5, 6]
    },
    # Update dict with Decision Tree Classifier
    "DTC": { 
            "classifier__criterion" :["gini", "entropy"],
            "classifier__splitter": ["best", "random"],
            "classifier__class_weight": [None, "balanced"],
            "classifier__max_features": [ "sqrt", "log2"],
            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
    },
    # randomized decision trees: a.k.a. extra-trees
    "ETC": {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__splitter": ["best", "random"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__max_depth": [1, 3, 5, 7, 8],
        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10]
    }
}

print('Hyperparameters Grid completed')

###############################################################################
#                      Classifier Tuning and Evaluation                  #
###############################################################################
# Initialize dictionary to store results
results = {}

# Tune and evaluate classifiers
for classifier_label, classifier in classifiers.items():
    try:       

        print(f"Now tuning {classifier_label}.")

       # Scale features via Z-score normalization
        scaler = StandardScaler()
        
        # Define steps in pipeline
        steps = [("scaler", scaler), ("classifier", classifier)]
        
        # Initialize Pipeline object
        pipeline_search = Pipeline(steps = steps)
        
        # Define parameter grid
        param_grid = parameters[classifier_label]
        
        # Initialize GridSearch object
        gscv = GridSearchCV(pipeline_search, param_grid, cv = 5,  verbose = 1, scoring = "roc_auc", n_jobs = -1)

        # Fit gscv and evaluate
        gscv.fit(X_train, np.ravel(y_train))  

        # Gets best params
        best_params = gscv.best_params_
        
        # Update classifier parameters 
        tuned_params = {item[12:]: best_params[item] for item in best_params}
        classifier.set_params(**tuned_params) # 0verriding previous parameter values passed 

        # Define new pipeline with tuned classifier
        steps = [("scaler", scaler), ("classifier", classifier)]
        pipeline = Pipeline(steps = steps)    

        # Start timer        
        start_time = time.time()

        # Re-fit the pipeline on the entire training set
        pipeline.fit(X_train, np.ravel(y_train))

        # End timer
        end_time = time.time()
        time_taken = (end_time - start_time) / 60  # Convert seconds to minutes
        print(f"Time taken for {classifier_label}: {time_taken:.2f} minutes.")

        # Make predictions using the pipeline (which includes the scaler)
        if classifier_label in DECISION_FUNCTIONS:
            y_pred_train = pipeline.decision_function(X_train)
            y_pred_test = pipeline.decision_function(X_test)
        else:
            y_pred_train = pipeline.predict_proba(X_train)[:, 1]
            y_pred_test = pipeline.predict_proba(X_test)[:, 1]

        # Score on training data
        train_auc = metrics.roc_auc_score(y_train, y_pred_train)

        # Score on test data
        test_auc = metrics.roc_auc_score(y_test, y_pred_test)
        
        # Save results
        results[classifier_label] = {
            "Best Parameters": str(best_params),
            "Training AUC": train_auc,
            "Test AUC": test_auc,
            "Time Taken (minutes)": time_taken
        }

    except Exception as e:
        print(f"Error with classifier {classifier_label}: {e}")
        results[classifier_label] = {"Error": str(e)}

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame.from_dict(results, orient='index')
filename = f"{current_dir}classifiers_AUC"

try:
    # Try to save as CSV
    results_df.to_csv(f"{filename}.csv")
    print(f"Results saved as CSV in {filename}.csv")
except Exception as e:
    print(f"Failed to save as CSV due to: {e}\nSaving as a TXT file.")

    # Save as TXT
    with open(f"{filename}.txt", "w") as file:
        for classifier_label, result in results.items():
            file.write(f"{classifier_label}:\n")
            for key, value in result.items():
                file.write(f"  {key}: {value}\n")
            file.write("\n")

    print(f"Results saved as TXT in {filename}.txt")

###############################################################################
#                              14. Visualing Results                          #
###############################################################################
# Initialize auc_score dictionary
auc_scores = {
              "Classifier": [],
              "AUC": [],
              "AUC Type": []
              }

# Get AUC scores into dictionary
for classifier_label in results:
    auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                       "AUC": [results[classifier_label]["Training AUC"]] + auc_scores["AUC"],
                       "AUC Type": ["Training"] + auc_scores["AUC Type"]})
    
    auc_scores.update({"Classifier": [classifier_label] + auc_scores["Classifier"],
                       "AUC": [results[classifier_label]["Test AUC"]] + auc_scores["AUC"],
                       "AUC Type": ["Test"] + auc_scores["AUC Type"]})

# Dictionary to PandasDataFrame
auc_scores = pd.DataFrame(auc_scores)
filename = current_dir + 'auc_scores.csv'
auc_scores.to_csv(filename)

# Function to plot AUC scores
def plot_auc_scores(auc_scores, current_dir):
    try:
        # Set graph style
        sns.set(font_scale=1.75, style="whitegrid")

        # Define colors
        training_color = sns.color_palette("RdYlBu", 10)[1]
        test_color = sns.color_palette("RdYlBu", 10)[-2]
        colors = [training_color, test_color]

        # Create bar plot
        f, ax = plt.subplots(figsize=(12, 9))
        sns.barplot(x="AUC", y="Classifier", hue="AUC Type", palette=colors, data=auc_scores)

        # Generate a bolded horizontal line at x = 0
        ax.axvline(x=0, color='black', linewidth=4, alpha=0.7)

        # Tight layout and save figure
        plt.tight_layout()
        filename = f"{current_dir}_AUC_Scores.png"
        plt.savefig(filename, dpi=1080)
        plt.close(f)  # Close the figure to free memory
    except Exception as e:
        print(f"An error occurred: {e}")

plot_auc_scores(auc_scores, current_dir)

