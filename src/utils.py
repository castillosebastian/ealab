import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import yaml
import os
from scipy.io import arff
import polars as pl

# Auxiliary functions

def bin(p=0.9):
    '''
    Esta función genera un bit al azar.
    '''
    if random.random() < p:
        return 1
    else:
        return 0

def mutation(ind, p):
    '''
    Esta función recorre el cromosoma y evalúa, para cada gen,
    si debe aplicar el operador de mutación.
    '''
    return [abs(i-1) if random.random() < p else i for i in ind]

def load_params(exp_path):
    with open(exp_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

# load data------
def load_data(train_file_name, test_file_name):
    # Get the directory of the current script (which is in the 'src' folder)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the root directory by going up one level from the 'src' folder
    root_dir = os.path.dirname(src_dir)
    
    # construc path
    train_path = os.path.join(root_dir, 'data', train_file_name)
    test_path = os.path.join(root_dir, 'data', test_file_name)
    
    try:
        # Load the ARFF files
        tra, _ = arff.loadarff(train_path)
        tst, _ = arff.loadarff(test_path)
        
        # Convert to Polars DataFrame and cast 'CLASS' column to string
        train = pl.DataFrame(tra).with_columns(pl.col('CLASS').cast(pl.datatypes.Utf8))
        test = pl.DataFrame(tst).with_columns(pl.col('CLASS').cast(pl.datatypes.Utf8))
        
        return train, test
    
    except FileNotFoundError:
        print(f"Error: File not found. Please ensure that '{train_file_name}' and '{test_file_name}' exist in the 'data' {train_path}.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def process_data(TRAIN, TEST, classname, scale=False):
    try:
        # Convert datasets to numpy arrays
        TRAIN = TRAIN.to_numpy()
        TEST = TEST.to_numpy()

        # Split features and labels
        X_TRAIN = TRAIN[:, :-1]
        y_train = TRAIN[:, -1]
        y_train = np.where(np.array(y_train) == classname, 1, 0).astype('int64')

        X_TEST = TEST[:, :-1]
        y_test = TEST[:, -1]
        y_test = np.where(np.array(y_test) == classname, 1, 0).astype('int64')

        # Scale the data if the scale parameter is True
        if scale:
            scaler = StandardScaler()
            scaler.fit(X_TRAIN)
            X_TRAIN = scaler.transform(X_TRAIN)
            X_TEST = scaler.transform(X_TEST)

        return X_TRAIN, y_train, X_TEST, y_test

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None


def fitness(features, Xtrain, Xtest, y_train, y_test):
        '''
        Función de aptitud empleada por nuestro algoritmo.
        '''
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if not isinstance(features[0], bool):
            features = features.astype(bool)
        
        X_train = Xtrain[:,features]
        X_test = Xtest[:,features]
        
        mlp = MLPClassifier(hidden_layer_sizes=(5,3),
                            activation='tanh',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate_init=0.001,
                            shuffle=True,
                            momentum=0.9,
                            validation_fraction=0.2,
                            n_iter_no_change=10,
                            random_state=42,
                            max_iter=3000).fit(X_train, y_train)

        yp = mlp.predict(X_test)

        acc = (y_test == yp).sum()/len(y_test)

        n_genes = 1 - (features.sum() / len(features))

        alpha = 0.5
        f = (alpha*acc + (1-alpha)*n_genes)
        
        return f, acc, n_genes