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
# Method to find the current directory and the root directory (assuming .git is in the root)
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
# Example usage:
current_dir, root_dir = find_dirs()
# If you need to append the root directory to sys.path:
if root_dir is not None:
    import sys
    if root_dir not in sys.path:
        sys.path.append(root_dir)

# params
experiment_name = "leukemia_base_001"
train_dir = root_dir + "/data/leukemia_train_38x7129.arff"
test_dir = root_dir + "/data/leukemia_test_34x7129.arff"
POP_SIZE = 100          # Cantidad de individuos en la población
PROB_MUT = 20.0         # Probabilidad de mutacion
PX = 0.75               # Probabilidad de cruza
GMAX = 10               # Cantidad máxima de generaciones que se ejecutará el algoritmo

# execution------------------------------------------------------------ 
# Start an MLflow experiment
mlflow.set_experiment(experiment_name)
# Set the tracking URI to a central directory

tra, trameta = arff.loadarff(train_dir)
tst, tstmeta = arff.loadarff(test_dir)
train = pl.DataFrame(tra)
train = train.with_columns(pl.col("CLASS").cast(pl.datatypes.Utf8))
test = pl.DataFrame(tst)
test = test.with_columns(pl.col("CLASS").cast(pl.datatypes.Utf8))

# recoding
TRAIN = train
TEST = test
TRAIN = TRAIN.to_numpy()
TEST = TEST.to_numpy()
X_TRAIN = TRAIN[:, :-1]
y_train = TRAIN[:, -1]
y_train = np.where(np.array(y_train) == "ALL", 1, 0).astype("int64")
X_TEST = TEST[:, :-1]
y_test = TEST[:, -1]
y_test = np.where(np.array(y_test) == "ALL", 1, 0).astype("int64")
scaler.fit(X_TRAIN)
Xtrain = scaler.transform(X_TRAIN)
Xtest = scaler.transform(X_TEST)

# cromosoma==========================
IND_SIZE = Xtrain.shape[1]  # Cantidad de genes en el cromosoma
PM = PROB_MUT / IND_SIZE    # Probabilidad de mutación [aproximadamente 1 gen por cromosoma]
                            # Experimento 4: con mayor probabilidad de mutación.
                            # PM = 20./IND_SIZE __experimento 2 mejoró el fitness y acc en 
                            # la segunda generación pero luego se estancó
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
def plot_evolution(logbook, chapter, y_label,filename=None, N_override=None, current_dir = None, experiment_name=experiment_name):
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
    N = N_override if N_override else (30 if GMAX > 200 else GMAX)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    generations = range(N)
    ax.plot(generations, avg[:N], "-or", label="Average")
    ax.plot(generations, max_[:N], "-og", label="Maximum")
    ax.plot(generations, min_[:N], "-ob", label="Minimum")
    ax.set_xlabel("Generations", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.legend(loc="best")
    ax.grid(True)

    filename = experiment_name +'_'+ filename

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


# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "POP_SIZE": POP_SIZE,
        "PROB_MUT": PROB_MUT,
        "PX": PX,
        "GMAX": GMAX,
    })

    # CREAMOS LA FUNCION DE FITNESS
    # Esta función tiene "1 OBJETIVO" a "MAXIMIZAR"
    creator.create(
        "Fitness",  # Nombre con el que se registra el componente
        base.Fitness,  # Clase de la que hereda
        weights=(1.0,),
    )
    # CREAMOS EL CONSTRUCTOR DE INDIVIDUOS
    creator.create(
        "Individual",  # Nombre con el que se registra el componente
        list,  # Clase de la que hereda [Tipo de contenedor en este caso]
        fitness=creator.Fitness,
        acc=0,
        ngenes=0,
    )  # Asignamos un método para evaluar el fitness del individuo

    # REGISTRAMOS COMPONENTES
    toolbox = base.Toolbox()
    # ---------------------
    # DEFINIMOS COMO CONSTRUIR UN GEN
    # el algoritmo retiene la historia de fitnes de genes activos, contribuyendo !!!IMPORTANTE
    # a la selección de las variables que contribuyen a mejorar el fitness
    toolbox.register("attribute", bin, p=0.1)  # Nombre con el que se registra el componente
    # Probabilidad de un "1":   exp1:0.1 =ELITE -- Fitness: 0.9513 -- NGENES: 694 -- Acc: 1.0
    #                           exp2:0.2 =ELITE -- Fitness: 0.8885 -- NGENES: 1380 -- Acc: 0.9706
    #                           exp3:0.05=ELITE -- Fitness: 0.9595 -- NGENES: 368 -- Acc: 0.9706
    #                           exp5:0.05 + 20/caracteristicas mutacion: ELITE -- Fitness: 0.9757 -- NGENES: 346 -- Acc: 1.0
    # ---------------------
    # DEFINIMOS COMO CONSTRUIR UN INDIVIDUO/CROMOSOMA
    toolbox.register(
        "individual",  # Nombre con el que se registra el componente
        tools.initRepeat,  # Método usado para construir el cromosoma
        creator.Individual,  # ...
        toolbox.attribute,  # Función para construir cada gen
        n=IND_SIZE,
    )  # Número de genes del cromosoma/individuo (se repetirá la función construir gen)
    # DEFINIMOS COMO CONSTRUIR LA POBLACION
    toolbox.register(
        "population",  # Nombre con el que se registra el componente
        tools.initRepeat,  # Método usado para construir el cromosoma
        list,
        toolbox.individual,
    )
    # DEFINIMOS COMO REALIZAR LA CRUZA
    toolbox.register(
        "mate", tools.cxTwoPoint  # Nombre con el que se registra el componente
    )  #
    # DEFINIMOS COMO REALIZAR LA MUTACION
    toolbox.register(
        "mutate",  # Nombre con el que se registra el componente
        mutation,  # Método de mutación (definido como función más arriba)
        p=PM,
    )  # Parámetro que usa la mutación
    # Otra opciones
    # toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    # DEFINIMOS COMO REALIZAR LA SELECCION DE INDIVIDUOS
    toolbox.register(
        "select",  # Nombre con el que se registra el componente
        tools.selTournament,  # Método usado para selección [selRoulette | selTournament | ...]
        tournsize=5,
    )  # Parámetro que usa el torneo

    # ## Definimos las estadísticas a calcular
    # EXTRAEMOS EL FITNESS DE TODOS LOS INDIVIDUOS
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    # EXTRAEMOS EL ACC DE TODOS LOS INDIVIDUOS
    stats_acc = tools.Statistics(key=lambda ind: ind.acc)
    # EXTRAEMOS LA FRACCION DE GENES ACTIVOS DE TODOS LOS INDIVIDUOS
    stats_frac_active_genes = tools.Statistics(key=lambda ind: ind.ngenes)
    # EXTRAEMOS EL NUMERO DE GENES ACTIVOS DE TODOS LOS INDIVIDUOS
    stats_active_genes = tools.Statistics(key=lambda ind: np.sum(ind))
    mstats = tools.MultiStatistics(
        fitness=stats_fit,
        acc=stats_acc,
        frac_ngenes=stats_frac_active_genes,
        ngenes=stats_active_genes,
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # INICIALIZAMOS UN LOGGER
    logbook = tools.Logbook()

    # Corremos el modelo con todas las features
    f = fitness(np.ones(Xtrain.shape[1]), Xtrain, Xtest, y_train, y_test)
    print(f"All features: FITNESS: {f[0]:.4} -- NGENES: {int(Xtrain.shape[1])} -- Acc: {f[1]:.4}\n")

    # ================================================
    # INICIALIZAMOS LA POBLACIÓN
    # ================================================
    pop = toolbox.population(n=POP_SIZE)  # Inicializamos una población
    # ================================================
    # EVALUAMOS EL FITNESS DE LA POBLACION
    # ======================================
    # fitnesses = list(map(toolbox.evaluate, pop))
    fitnesses = Parallel(n_jobs=4, backend="multiprocessing")(
        delayed(fitness)(ind, Xtrain, Xtest, y_train, y_test) for ind in pop
    )
    # ================================================
    # ASIGNAMOS A CADA INDIVIDUO SU FITNESS
    # ========================================
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (
            fit[0],
        )  # Guardamos el fitness para cada individuo (en el individuo)
        ind.acc = fit[1]
        ind.ngenes = fit[2]
    # ================================================
    records = mstats.compile(pop)
    logbook.record(gen=0, **records)
    # ================================================
    # COMENZAMOS LA EVOLUCION
    # ========================================
    for g in range(1, GMAX):  
        # ================================================
        # SELECCIONAMOS INDIVIDUO ELITE
        # ================================
        idx_elite = np.argmax(fitnesses, axis=0)[
            0
        ]  # Si maximizamos, hay que usar ".argmax()". El indiv. con mejor fitness
        elite = toolbox.clone(pop[idx_elite])
        del elite.fitness.values, elite.acc, elite.ngenes
        # ================================================
        # HACEMOS UNA COPIA DE LA POBLACION ACTUAL
        # ==========================================
        parents = toolbox.select(pop, POP_SIZE)  # Seleccionamos individuos para alcanzar
        # el tamaño de la población
        offspring = list(map(toolbox.clone, pop))  # Clonamos para tener nuevos individuos
        # (contenedores independientes)
        # ============================================
        # REALIZAMOS LA CRUZA DE LOS PADRES
        # ============================================
        for i in range(POP_SIZE // 2):
            parent1 = toolbox.clone(parents[random.randint(0, POP_SIZE - 1)])
            parent2 = toolbox.clone(parents[random.randint(0, POP_SIZE - 1)])

            if random.random() < PX:
                childs = toolbox.mate(parent1, parent2)
            else:
                childs = (parent1, parent2)

            offspring[2 * i] = childs[0]
            offspring[2 * i + 1] = childs[1]
        # ================================================
        # MUTAMOS LOS HIJOS
        # =================================
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values  # , mutant.acc, mutant.ngenes
        # ================================================
        # EVALUAMOS EL FITNESS Y SE LO ASIGNAMOS A CADA INDIVIDUO
        # ======================================
        offspring[0] = elite
        fitnesses = Parallel(n_jobs=4, backend="multiprocessing")(
            delayed(fitness)(ind, Xtrain, Xtest, y_train, y_test) for ind in offspring
        )
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (
                fit[0],
            )  # Guardamos el fitness para cada individuo (en el individuo)
            ind.acc = fit[1]
            ind.ngenes = fit[2]
        # ================================================
        # CONSTRUIMOS LA NUEVA POBLACION
        # ================================
        pop = toolbox.clone(offspring)
        # ================================================
        # CALCULAMOS ESTADÏSTICAS
        # ============================
        records = mstats.compile(pop)
        logbook.record(gen=g, **records)
        if g % 1 == 0:
            print("=" * 79)
            print(f"GENERATION: {g}")
            print(
                f"Elite -- Fitness: {elite.fitness.values[0]:.4} -- NGENES: {np.sum(elite)} -- Acc: {elite.acc:.4}"
            )
            print("Poblacion FITNES: ", records["fitness"])
            print("Poblacion ACC: ", records["acc"])
            print("Poblacion GENES: ", records["ngenes"])
        # ================================================

    # After the genetic algorithm finishes, log the final population metrics
    final_records = mstats.compile(pop)
    mlflow.log_metrics({
        "final_fitness_avg": final_records['fitness']['avg'],
        "final_accuracy_avg": final_records['acc']['avg'],
        "final_ngenes_avg": final_records['ngenes']['avg'],
    })

    # Generate and log plots as artifacts
    fitness_plot_path = plot_evolution(logbook, "fitness", "Fitness", current_dir=current_dir, filename="fitness_evolution.png")
    acc_plot_path = plot_evolution(logbook, "acc", "Accuracy", current_dir=current_dir, filename="accuracy_evolution.png")
    genes_plot_path = plot_evolution(logbook, "ngenes", "Number of Genes", filename="genes_evolution.png", current_dir=current_dir, N_override=100 if GMAX > 2000 else GMAX)

    if fitness_plot_path:
        mlflow.log_artifact(fitness_plot_path, "plots")
    if acc_plot_path:
        mlflow.log_artifact(acc_plot_path, "plots")
    if genes_plot_path:
        mlflow.log_artifact(genes_plot_path, "plots")