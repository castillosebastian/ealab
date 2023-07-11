import numpy as np
import random
from sklearn.neural_network import MLPClassifier
import yaml

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