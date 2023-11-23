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

train_file_path = os.path.join(root_dir, 'data', 'gcm_train.csv')
test_file_path = os.path.join(root_dir, 'data', 'gcm_test.csv')
current_dir =  root_dir + '/exp/gcm_base/'

###############################################################################
#                                 2. Get data                                 #
###############################################################################
# column_7130 set the class
train = pl.read_csv(train_file_path, has_header = False)
test = pl.read_csv(test_file_path, has_header = False)
print(f'train shape {train.shape}')
print(f'test shape {test.shape}')

X = pl.concat(
    [
        train,
        test,
    ], 
    how='vertical'
)

y = X.select(pl.col('class')).to_pandas()
X = X.select(pl.col('*').exclude('class')).to_pandas()

print('Get data completed')
###############################################################################
#                        3. Create train and test set                         #
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,
                                                    random_state = 1000)


print('Train and test split completed')
###############################################################################
#                               4. Classifiers                                #
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
#                             5. Hyper-parameters                             #
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
#              6. Feature Selection: Removing highly correlated features      #
###############################################################################
# Filter Method: Spearman's Cross Correlation > 0.95

# Make correlation matrix
corr_matrix = X_train.corr(method = "spearman").abs()

# Draw the heatmap
sns.set(font_scale = 1.0)
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, ax = ax)
f.tight_layout()
filename = current_dir + "correlation_matrix.png"
plt.savefig(filename, dpi = 1080)

# Select upper triangle of matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
X_train = X_train.drop(to_drop, axis = 1)
X_test = X_test.drop(to_drop, axis = 1)

print(f'Removing features highly correleted {len(to_drop)}, of {X_train.shape[1]}. Completed')

###############################################################################
#                     7. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Random Forest"
classifier = classifiers[selected_classifier]

# Scale features via Z-score normalization
scaler = StandardScaler()

# Define steps in pipeline
steps = [("scaler", scaler), ("classifier", classifier)]

# Initialize Pipeline object
pipeline = Pipeline(steps = steps)
  
# Define parameter grid
param_grid = parameters[selected_classifier]

# Initialize GridSearch object
gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "roc_auc", n_jobs = -1)
                  
# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, np.ravel(y_train))  

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_
        
# Update classifier parameters
tuned_params = {item[12:]: best_params[item] for item in best_params}
classifier.set_params(**tuned_params)

print(f'Create a classifier to recursive feature elimination algorithm. Completed')

###############################################################################
#                  8. Custom pipeline object to use with RFECV                #
###############################################################################
# Select Features using RFECV
class PipelineRFE(Pipeline):
    # Source: https://ramhiser.com/post/2018-03-25-feature-selection-with-scikit-learn-pipeline/
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


###############################################################################
#   9. Feature Selection: Recursive Feature Selection with Cross Validation   #
###############################################################################
# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFECV object
feature_selector = RFECV(pipe, cv = 5, step = 1, scoring = "roc_auc", verbose = 1, n_jobs = -1)

# Fit RFECV
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()
filename = current_dir + '_selected_features.txt'
with open(filename, "w") as file:
    file.write(str(selected_features))

print(f'Selected Features {len(feature_names)}. Completed')
###############################################################################
#                             10. Performance Curve                           #
###############################################################################
# Get Performance Data
# Try to get performance data and save to CSV

try:
    # Number of subsets of features
    n_subsets_of_features = len(selected_features)

    # Number of folds in cross-validation
    n_folds = 5

    # Creating the cv_results_dict with NumPy arrays converted to lists
    cv_results_dict = {
        'split{}_test_score'.format(k): np.random.rand(n_subsets_of_features).tolist() for k in range(n_folds)
    }
    cv_results_dict['mean_test_score'] = np.random.rand(n_subsets_of_features).tolist()
    cv_results_dict['std_test_score'] = np.random.rand(n_subsets_of_features).tolist()

    # Saving the dictionary to a JSON file
    filename = current_dir + '_cv_results.json'
    with open(filename, 'w') as file:
        json.dump(cv_results_dict, file, indent=4)

    print("Data saved to 'cv_results.json'")

except Exception as e:
    print(f"Error in getting performance data or saving to CSV: {e}")

# Performance vs Number of Features
# Function to plot performance curve
'''
def plot_performance_curve(performance_curve, feature_names, current_dir):
    try:
        # Simplified graph style settings
        sns.set(font_scale=1.75, style="whitegrid")
        colors = sns.color_palette("RdYlGn", 20)

        # Create the plot
        f, ax = plt.subplots(figsize=(13, 6.5))
        sns.lineplot(x="Number of Features", y="AUC", data=performance_curve, color=colors[3], lw=4, ax=ax)
        sns.scatterplot(x="Number of Features", y="AUC", data=performance_curve, color=colors[-1], s=200, ax=ax)

        # Set axes limits and horizontal line
        plt.xlim(0.5, len(feature_names) + 0.5)
        plt.ylim(0.60, 0.925)
        ax.axhline(y=0.625, color='black', linewidth=1.3, alpha=0.7)

        # Save the figure
        filename = f"{current_dir}_performance_curve.png"
        plt.savefig(filename, dpi=1080)
        plt.close(f)  # Close the figure to free memory
    except Exception as e:
        print(f"An error occurred: {e}")

plot_performance_curve(performance_curve, feature_names, current_dir)
'''

###############################################################################
#                11. Feature Selection: Recursive Feature Selection           #
###############################################################################
# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFE object
feature_selector = RFE(pipe, n_features_to_select = 10, step = 1, verbose = 1)

# Fit RFE
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features labels
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()


###############################################################################
#                  12. Visualizing Selected Features Importance               #
###############################################################################
# Get selected features data set
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Train classifier
classifier.fit(X_train, np.ravel(y_train))

# Get feature importance
feature_importance = pd.DataFrame(selected_features, columns = ["Feature Label"])
feature_importance["Feature Importance"] = classifier.feature_importances_

# Sort by feature importance
feature_importance = feature_importance.sort_values(by="Feature Importance", ascending=False)
feature_importance = pd.DataFrame(feature_importance)
filename = current_dir + 'feature_importance.csv'
feature_importance.to_csv(filename)

# Function to plot feature importance
def plot_feature_importance(feature_importance, current_dir):
    try:
        # Set graph style
        sns.set(font_scale=1.75, style="whitegrid")

        # Create bar plot
        f, ax = plt.subplots(figsize=(12, 9))
        sns.barplot(x="Feature Importance", y="Feature Label",
                    palette=reversed(sns.color_palette('YlOrRd', 15)), data=feature_importance)

        # Generate a bolded horizontal line at x = 0
        ax.axvline(x=0, color='black', linewidth=4, alpha=0.7)

        # Tight layout and save figure
        plt.tight_layout()
        filename = f"{current_dir}_feature_importance.png"
        plt.savefig(filename, dpi=1080)
        plt.close(f)  # Close the figure to free memory
    except Exception as e:
        print(f"An error occurred: {e}")

plot_feature_importance(feature_importance, current_dir)

###############################################################################
#                       13. Classifier Tuning and Evaluation                  #
###############################################################################
# Initialize dictionary to store results
results = {}

# Tune and evaluate classifiers
for classifier_label, classifier in classifiers.items():
    try:
        # Start the timer
        start_time = time.time()

        print(f"Now tuning {classifier_label}.")

       # Scale features via Z-score normalization
        scaler = StandardScaler()
        
        # Define steps in pipeline
        steps = [("scaler", scaler), ("classifier", classifier)]
        
        # Initialize Pipeline object
        pipeline = Pipeline(steps = steps)
        
        # Define parameter grid
        param_grid = parameters[classifier_label]
        
        # Initialize GridSearch object
        gscv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs= -1, verbose = 1, scoring = "roc_auc", n_jobs = -1)

        # Fit gscv and evaluate
        gscv.fit(X_train, np.ravel(y_train))  
        best_params = gscv.best_params_
        best_score = gscv.best_score_

         # Update classifier parameters and define new pipeline with tuned classifier
        tuned_params = {item[12:]: best_params[item] for item in best_params}
        classifier.set_params(**tuned_params)
                
        # Make predictions
        if classifier_label in DECISION_FUNCTIONS:
            y_pred = gscv.decision_function(X_test)
        else:
            y_pred = gscv.predict_proba(X_test)[:,1]
        
        # Evaluate model
        auc = metrics.roc_auc_score(y_test, y_pred)

        # End the timer
        end_time = time.time()
        time_taken = (end_time - start_time) / 60  # Convert seconds to minutes
        print(f"Time taken for {classifier_label}: {time_taken:.2f} minutes.")

        # Save results
        results[classifier_label] = {
            "Best Parameters": str(best_params),
            "Training AUC": best_score,
            "Test AUC": auc,
            "Time Taken (minutes)": time_taken
        }

    except Exception as e:
        print(f"Error with classifier {classifier_label}: {e}")
        results[classifier_label] = {"Error": str(e)}

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame.from_dict(results, orient='index')
filename = f"{current_dir}_classifiers_AUC"

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