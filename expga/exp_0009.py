import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from deap import (base)  # Estructura que permite agrupar todos los componentes de nuestro algoritmo en una misma bolsa
from deap import creator  # Permite crear los componentes de nuestro algoritmo
from deap import tools  # Contiene funciones precargadas
from joblib import Parallel, delayed
scaler = StandardScaler()
import os
import sys
# Method to find the root directory (assuming .git is in the root)
def find_root_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # To avoid infinite loop
        if ".git" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None  # Or raise an error if the root is not found
root = find_root_dir()
sys.path.append(root)
from src.ga_base import *
import dagshub
dagshub.init(repo_owner='castilloclaudiosebastian', repo_name='ealab', mlflow=True)


# params
experiment_name = "leukemia_base_0009"
description = "Set up metrics"
current_dir = root +  "/expga/"
train_dir = root + "/data/leukemia_train_38x7129.arff"
test_dir = root + "/data/leukemia_test_34x7129.arff"
POP_SIZE = 100          # Cantidad de individuos en la población
PROB_MUT = 1        # Probabilidad de mutacion
PX = 0.75               # Probabilidad de cruza
GMAX = 100               # Cantidad máxima de generaciones que se ejecutará el algoritmo


Xtrain, y_train, Xtest, y_test = load_and_preprocess_data(train_dir=train_dir, test_dir=test_dir,
                                                            class_column_name='CLASS', 
                                                            class_value_1='ALL')

IND_SIZE = Xtrain.shape[1]  # Cantidad de genes en el cromosoma
PM = PROB_MUT / IND_SIZE    # Probabilidad de mutación [aproximadamente 1 gen por cromosoma]
                            # Experimento 4: con mayor probabilidad de mutación.
                            # PM = 20./IND_SIZE __experimento 2 mejoró el fitness y acc en 
                            # la segunda generación pero luego se estancó
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id, run_name=experiment_name) as run:
    mlflow.set_tag("description", description)

    # Log parameters
    mlflow.log_params({
        "POP_SIZE": POP_SIZE,
        "PROB_MUT": PM,
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

        file_path = f"{current_dir}/{experiment_name}.txt"
        with open(file_path, 'a') as file:
            if g % 1 == 0:
                file.write("=" * 79 + "\n")
                file.write(f"GENERATION: {g}\n")
                file.write(
                    f"Elite -- Fitness: {elite.fitness.values[0]:.4} -- NGENES: {np.sum(elite)} -- Acc: {elite.acc:.4}\n"
                )
                file.write("Poblacion FITNES: " + str(records["fitness"]) + "\n")
                file.write("Poblacion ACC: " + str(records["acc"]) + "\n")
                file.write("Poblacion GENES: " + str(records["ngenes"]) + "\n")
                file.write("#" * 79 + "\n")
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
        "pob_fitness_avg": final_records['fitness']['avg'],
        "pob_accuracy_avg": final_records['acc']['avg'],
        "pob_ngenes_avg": final_records['ngenes']['avg'],
        "pob_fitness_max": final_records['fitness']['max'],
        "pob_accuracy_max": final_records['acc']['max'],
        "pob_ngenes_max": final_records['ngenes']['max'],
         
    })

    # Generate and log plots as artifacts
    fitness_plot_path = plot_evolution(logbook, "fitness", "Fitness", 
                                       current_dir=current_dir, 
                                       experiment_name=experiment_name,
                                       filename="fitness_evolution.png", 
                                       GMAX = GMAX)
    acc_plot_path = plot_evolution(logbook, "acc", "Accuracy", 
                                   current_dir=current_dir, 
                                   experiment_name=experiment_name,
                                   filename="accuracy_evolution.png", 
                                   GMAX = GMAX)
    genes_plot_path = plot_evolution(logbook, "ngenes", "Number of Genes",
                                    experiment_name=experiment_name,
                                    filename="genes_evolution.png", 
                                    current_dir=current_dir, 
                                    N_override= GMAX)

    if fitness_plot_path:
        mlflow.log_artifact(fitness_plot_path, "plots")
    if acc_plot_path:
        mlflow.log_artifact(acc_plot_path, "plots")
    if genes_plot_path:
        mlflow.log_artifact(genes_plot_path, "plots")


    mlflow.end_run()