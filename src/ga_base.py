import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import mlflow
import mlflow.sklearn
from tqdm import tqdm
from deap import (base)  # Estructura que permite agrupar todos los componentes de nuestro algoritmo en una misma bolsa
from deap import creator  # Permite crear los componentes de nuestro algoritmo
from deap import tools  # Contiene funciones precargadas
from joblib import Parallel, delayed
from scipy.io import arff
scaler = StandardScaler()
import os
import inspect


def find_dirs():
    # Fallback for environments where __file__ is not defined
    fallback_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))        
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = fallback_dir    
    root_dir = current_dir  # Start searching from the current directory
    while root_dir != os.path.dirname(root_dir):  # To avoid infinite loop
        if ".git" in os.listdir(root_dir):
            return current_dir, root_dir  # Return both current and root directories
        root_dir = os.path.dirname(root_dir)    
    return current_dir, None  # Return current directory and None if root is not found

def load_and_preprocess_data(train_dir, test_dir, class_column_name=None, class_value_1=None, selected_features=False):  
    # Load the data
    tra, _ = arff.loadarff(train_dir)
    tst, _ = arff.loadarff(test_dir)
    
    # Convert to pandas DataFrames
    train = pd.DataFrame(tra)
    test = pd.DataFrame(tst)

    if selected_features:
        # read selected features from file /home/sebacastillo/ealab/data/features_occurrence.csv
        selected_features = pd.read_csv('/home/sebacastillo/ealab/data/features_occurrence.csv')
        # order the features by occurrence
        selected_features = selected_features.sort_values(by='count', ascending=False)
        # select the top 30 features
        selected_features = selected_features.head(30)
        selected_features = selected_features['selected_features_bin'].tolist()
        # add the class column
        selected_features.append(class_column_name)
        train = train[selected_features]
        test = test[selected_features]

    # Get the column names
    features = list(train.columns)
    
    # Ensure the class label column is in string format for consistency
    train[class_column_name] = train[class_column_name].astype(str)
    test[class_column_name] = test[class_column_name].astype(str)
    
    # Recoding the class labels based on 'class_value'
    y_train = np.where(train[class_column_name] == class_value_1, 1, 0).astype("int64")
    y_test = np.where(test[class_column_name] == class_value_1, 1, 0).astype("int64")
    
    # Drop the class label column to isolate features
    X_train = train.drop(columns=[class_column_name])
    X_test = test.drop(columns=[class_column_name])
    
    # Initialize and fit scaler on training data
    scaler = StandardScaler().fit(X_train)
    
    # Scale the features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, features

# Funciones
# =================================
def bin(p=0.9):
    """
    Esta función genera un bit al azar.
    """
    if random.random() < p:
        return 1
    else:
        return 0
# =================================
def fitness(features, Xtrain, Xtest, y_train, y_test):
    """
    Función de aptitud empleada por nuestro algoritmo.
    """
    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if not isinstance(features[0], bool):
        features = features.astype(bool)

    X_train = Xtrain[:, features]
    X_test = Xtest[:, features]

    mlp = MLPClassifier(
        hidden_layer_sizes=(5, 3),
        activation="tanh",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        shuffle=True,
        momentum=0.9,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=42,
        max_iter=3000,
    ).fit(X_train, y_train)

    yp = mlp.predict(X_test)

    acc = (y_test == yp).sum() / len(y_test)

    n_genes = 1 - (features.sum() / len(features))

    alpha = 0.5

    f = alpha * acc + (1 - alpha) * n_genes

    return f, acc, n_genes
# =================================
def plot_evolution(logbook, chapter, y_label,filename=None, 
                   N_override=None, current_dir = None, experiment_name=None, GMAX = 100):
    """
    Plot the evolution of a given statistic (chapter) from the logbook.
    Parameters:
    - logbook: The DEAP logbook containing the statistics.
    - chapter: The name of the chapter in the logbook to plot (e.g., 'fitness', 'acc', 'ngenes').
    - y_label: The label for the y-axis.
    - N_override: Optional, override the number of generations to plot.
    """
    chapter_data = logbook.chapters[chapter]
    avg = chapter_data.select("avg")
    max_ = chapter_data.select("max")
    min_ = chapter_data.select("min")
    N = N_override if N_override else GMAX
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    generations = range(N)
    ax.plot(generations, avg[:N], "-or", label="Average")
    ax.plot(generations, max_[:N], "-og", label="Maximum")
    ax.plot(generations, min_[:N], "-ob", label="Minimum")
    ax.set_xlabel("Generations", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.legend(loc="best")
    ax.grid(True)

    filename = experiment_name + '_' + filename

    if filename:
        plot_path = os.path.join(current_dir, filename)
        plt.savefig(plot_path, format='png', dpi=80)
        plt.close()
        return plot_path
    return None

# ====================================
def mutation(ind, p):
    """
    Esta función recorre el cromosoma y evalúa, para cada gen,
    si debe aplicar el operador de mutación.
    """
    return [abs(i - 1) if random.random() < p else i for i in ind]


