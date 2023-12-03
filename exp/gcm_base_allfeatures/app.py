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
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, make_scorer
import time  # Import the time module
from sklearn.metrics import classification_report

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

train_file_path = os.path.join(root_dir, 'data', 'gcm_train.csv')
test_file_path = os.path.join(root_dir, 'data', 'gcm_test.csv')
current_dir =  root_dir + '/exp/gcm_base_allfeatures/'

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
#                               4. Classifiers                                #
###############################################################################
# Create list of tuples with classifier label and classifier object
# Update the classifiers dictionary for multi-class classification
# Selected classifiers based on the posibility of class weight balanced!
classifiers = { 
    "Gradient Boosting": GradientBoostingClassifier(),   
    "Extra Trees Ensemble": ExtraTreesClassifier(class_weight='balanced'),    
    "Random Forest": RandomForestClassifier(class_weight='balanced'),   
    "SGD": SGDClassifier(class_weight='balanced'),
    "LSVC": LinearSVC(class_weight='balanced'),
    "NuSVC": NuSVC(class_weight='balanced', decision_function_shape='ovo'),
    "SVC": SVC(class_weight='balanced', decision_function_shape='ovo'),
    "DTC": DecisionTreeClassifier(class_weight='balanced'),
    "ETC": ExtraTreeClassifier(class_weight='balanced'),
    "MLP": MLPClassifier(),
}

print('Classifiers completed')
###############################################################################
#                             5. Hyper-parameters                             #
###############################################################################
# Defining the hyperparameter spaces for the specified machine learning models
parameters = {
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

    "Extra Trees Ensemble": { 
        "classifier__n_estimators": [100, 200, 300],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__n_jobs": [-1]
    },

    "Random Forest": { 
        "classifier__n_estimators": [100, 200, 300],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__n_jobs": [-1]
    },

    "SGD": { 
        "classifier__loss": ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
        "classifier__alpha": [1e-7, 1e-5, 1e-3, 1e-2, 1e-1],
        "classifier__penalty": ['l2', 'l1', 'elasticnet'],
        "classifier__learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
        "classifier__eta0": [0.001, 0.01, 0.1],
        "classifier__n_jobs": [-1]
    },

    "LSVC": { 
        "classifier__penalty": ['l2'],
        "classifier__loss": ['hinge', 'squared_hinge'],
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__dual": [True, False],
        "classifier__max_iter": [1000, 2000, 5000]
    },
    "NuSVC": { 
        "classifier__nu": [0.25, 0.5, 0.75],
        "classifier__kernel": ['linear', 'rbf', 'poly'],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ['scale', 'auto'],
        "classifier__coef0": [0.0, 0.5, 1.0]
    },
    "SVC": { 
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__kernel": ['linear', 'rbf', 'poly'],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ['scale', 'auto'],
        "classifier__coef0": [0.0, 0.5, 1.0]
    },

    "DTC": { 
        "classifier__criterion": ['gini', 'entropy'],
        "classifier__splitter": ['best', 'random'],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ['sqrt', 'log2', None]
    },
    "ETC": { 
        "classifier__criterion": ['gini', 'entropy'],
        "classifier__splitter": ['random', 'best'],
        "classifier__max_depth": [None, 5, 10, 15],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ['sqrt', 'log2', None]
    },
    "MLP": { 
        "classifier__hidden_layer_sizes": [(50,40), (100,80), (50, 50), (100, 100)],
        "classifier__activation": ['identity', 'tanh', 'relu'],
        "classifier__solver": ['lbfgs', 'sgd', 'adam'],
        "classifier__alpha": [0.0001, 0.001, 0.01],
        "classifier__learning_rate": ['constant', 'invscaling', 'adaptive'],
        "classifier__max_iter": [200, 400, 800],
        "classifier__learning_rate_init": [0.001, 0.01, 0.1]
    }
}

print('Hyperparameters Grid completed')

###############################################################################
#                      Classifier Tuning and Evaluation                  #
###############################################################################
# Initialize dictionary to store results
results = {}
pipeline_out = {}
report_out = {}

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
        
        # Custom scorer for multi-class
        f1_weighted_scorer = make_scorer(f1_score, average='weighted')

        # Initialize RandomizedSearchCV object with a multi-class compatible scorer
        # Adjust n_iter to control the number of parameter settings sampled
        gscv = GridSearchCV(pipeline_search, param_grid, cv=5, n_jobs=-1, verbose=1, scoring=f1_weighted_scorer)  

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
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
                     
        # Evaluate model
        train_f1_score = f1_score(y_train, y_pred_train, average='weighted')
        train_accuracy = accuracy_score(y_train, y_pred_train)

        test_f1_score = f1_score(y_test, y_pred_test, average='weighted')
        test_accuracy = accuracy_score(y_test, y_pred_test)
                
        # Save results
        results[classifier_label] = {
            "Best Parameters": str(best_params),     
            "Training F1-Score": train_f1_score,    
            "Train Accuracy": train_accuracy,   
            "Test F1-Score": test_f1_score,
            "Test Accuracy": test_accuracy,
            "Time Taken (minutes)": time_taken
        }     

        pipeline_out[classifier_label] = {
            'output' : pipeline.named_steps['classifier'].__dict__
        }

        report_out[classifier_label] = {
            'output' : classification_report(y_test, y_pred_test, output_dict=True)
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

# Save objetcs
results_df = pd.DataFrame.from_dict(pipeline_out, orient='index')
filename = f"{current_dir}pipeline_output_obj"
# Generating classification report to save
report = pd.DataFrame.from_dict(report_out, orient='index')
filename_report = f"{current_dir}report_obj"

try:
    # Try to save as CSV
    results_df.to_csv(f"{filename}.csv")
    report.to_csv(f"{filename_report}.csv")

except Exception as e:
    print(f"Failed to save as CSV due to: {e}\nSaving as a TXT file.")



###############################################################################
#                              14. Visualing Results                          #
###############################################################################
# Initialize f1_score dictionary
f1_scores = {
    "Classifier": [],
    "F1 Score": [],
    "F1 Score Type": []
}

# Get F1 scores into the dictionary
for classifier_label in results:
    f1_scores["Classifier"].extend([classifier_label, classifier_label])
    f1_scores["F1 Score"].extend([results[classifier_label]["Training F1-Score"], results[classifier_label]["Test F1-Score"]])
    f1_scores["F1 Score Type"].extend(["Training", "Test"])

# Dictionary to Pandas DataFrame
f1_scores_df = pd.DataFrame(f1_scores)
filename = current_dir + 'f1_scores.csv'
f1_scores_df.to_csv(filename)

# Function to plot F1 scores
def plot_f1_scores(f1_scores_df, current_dir):
    try:
        # Set graph style
        sns.set(font_scale=1.75, style="whitegrid")

        # Define colors
        training_color = sns.color_palette("RdYlBu", 10)[1]
        test_color = sns.color_palette("RdYlBu", 10)[-2]
        colors = [training_color, test_color]

        # Create bar plot
        f, ax = plt.subplots(figsize=(12, 9))
        sns.barplot(x="F1 Score", y="Classifier", hue="F1 Score Type", palette=colors, data=f1_scores_df)

        # Generate a bolded horizontal line at x = 0
        ax.axvline(x=0, color='black', linewidth=4, alpha=0.7)

        # Tight layout and save figure
        plt.tight_layout()
        filename = f"{current_dir}_F1_Scores.png"
        plt.savefig(filename, dpi=1080)
        plt.close(f)  # Close the figure to free memory
    except Exception as e:
        print(f"An error occurred: {e}")

plot_f1_scores(f1_scores_df, current_dir)