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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, make_scorer
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
current_dir =  root_dir + '/exp/gcm_base_gscv/'

###############################################################################
#                                 2. Get data                                 #
###############################################################################
# column_7130 set the class
train = pl.read_csv(train_file_path, has_header = True)
test = pl.read_csv(test_file_path, has_header = True)
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

# Create dict of decision function labels
DECISION_FUNCTIONS = {"Ridge", "SGD", "LSVC", "NuSVC", "SVC"}

# Create dict for classifiers with feature_importances_ attribute
FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}

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
#                     7. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
'''
selected_classifier = "Gradient Boosting" # Random Fores did not work
classifier = classifiers[selected_classifier]

# Scale features via Z-score normalization
scaler = StandardScaler()

# Define steps in pipeline
steps = [("scaler", scaler), ("classifier", classifier)]

# Initialize Pipeline object
pipeline = Pipeline(steps=steps)

# Define parameter grid
param_grid = parameters[selected_classifier]

# Custom scorer for multi-class - here using weighted F1 score
f1_weighted_scorer = make_scorer(f1_score, average='weighted')

# Initialize GridSearch object with a multi-class compatible scorer
rscv = RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, n_jobs=-1, verbose=1, scoring=f1_weighted_scorer, random_state=42)

# Fit rscv
print(f"Now tuning {selected_classifier}. Please wait.")
rscv.fit(X_train, np.ravel(y_train))  

# Get best parameters and score
best_params = rscv.best_params_
best_score = rscv.best_score_

# Update classifier parameters
tuned_params = {item.replace("classifier__", ""): best_params[item] for item in best_params}
classifier.set_params(**tuned_params)

print(f'Classifier {selected_classifier} tuned and ready for feature elimination. Completed')


###############################################################################
#                  8. Custom pipeline object to use with RFECV                #
###############################################################################

# Custom pipeline class to handle feature importances

class PipelineRFE(Pipeline):
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'feature_importances_'):
            self.feature_importances_ = self.steps[-1][-1].feature_importances_
        elif hasattr(self.steps[-1][-1], 'coef_'):
            self.feature_importances_ = np.abs(self.steps[-1][-1].coef_[0])
        else:
            raise AttributeError("Selected classifier does not have feature_importances_ or coef_ attribute.")
        return self

# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps=steps)

# Custom scorer for multi-class - using weighted F1 score
f1_weighted_scorer = make_scorer(f1_score, average='weighted')

# Initialize RFECV object with a multi-class compatible scorer
feature_selector = RFECV(pipe, cv=5, step=1, scoring=f1_weighted_scorer, verbose=1, n_jobs=-1)

# Fit RFECV
print("Performing feature selection. Please wait.")
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()
filename = current_dir + '_selected_features.txt'
with open(filename, "w") as file:
    file.write(str(selected_features))

print(f'Selected Features: {len(selected_features)} out of {len(feature_names)}. Completed')

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


###############################################################################
#                11. Feature Selection: Recursive Feature Selection           #
###############################################################################

# Define pipeline for RFECV
steps = [("scaler", scaler), ("classifier", classifier)]
pipe = PipelineRFE(steps = steps)

# Initialize RFE object
feature_selector = RFE(pipe, n_features_to_select=10, step=1, verbose=1)

# Fit RFE
print("Performing feature selection using RFE. Please wait.")
feature_selector.fit(X_train, np.ravel(y_train))

# Get selected features labels
feature_names = X_train.columns
selected_features = feature_names[feature_selector.support_].tolist()

# Update datasets with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train classifier with selected features
classifier.fit(X_train_selected, np.ravel(y_train))

# Check if classifier has 'feature_importances_' or 'coef_' attribute
if hasattr(classifier, 'feature_importances_'):
    feature_importance_values = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    feature_importance_values = np.abs(classifier.coef_[0])
else:
    raise AttributeError("Classifier does not provide feature importance.")


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

plot_feature_importance(feature_importance, current_dir
'''
###############################################################################
#                       13. Classifier Tuning and Evaluation                  #
###############################################################################

# Initialize dictionary to store results
results = {}

# Tune and evaluate classifiers using Randomized Grid Search
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
        pipeline = Pipeline(steps=steps)

        # Define parameter grid
        param_grid = parameters[classifier_label]

        # Custom scorer for multi-class
        f1_weighted_scorer = make_scorer(f1_score, average='weighted')

        # Initialize RandomizedSearchCV object with a multi-class compatible scorer
        # Adjust n_iter to control the number of parameter settings sampled
        gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring=f1_weighted_scorer)       

        # Fit RandomizedSearchCV and evaluate
        gs.fit(X_train, np.ravel(y_train))
        best_params = gs.best_params_
        best_score = gs.best_score_

        # Update classifier parameters and define new pipeline with tuned classifier
        tuned_params = {item.replace("classifier__", ""): best_params[item] for item in best_params}
        classifier.set_params(**tuned_params)

        # Make predictions
        y_pred = gs.predict(X_test)
        
        # Evaluate model
        test_f1_score = f1_score(y_test, y_pred, average='weighted')
        test_accuracy = accuracy_score(y_test, y_pred)

        # End the timer
        end_time = time.time()
        time_taken = (end_time - start_time) / 60  # Convert seconds to minutes
        print(f"Time taken for {classifier_label}: {time_taken:.2f} minutes.")

        # Save results
        results[classifier_label] = {
            "Best Parameters": str(best_params),
            "Training F1-Score": best_score,
            "Test F1-Score": test_f1_score,
            "Test Accuracy": test_accuracy,
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