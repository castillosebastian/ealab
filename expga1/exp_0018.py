import random
import csv
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import json
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
from src.bo_vae2 import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# params
experiment_name = "mandelon_base_0018"
description = "3experiments_originaldata_001xgen"
current_dir = root +  "/expga1"
dataset_name = "madelon"
class_column = "class"
class_1 = "1"
train_dir = root + "/data/madelon.trn.arff"
test_dir = root + "/data/madelon.tst.arff"
POP_SIZE = 100          # Cantidad de individuos en la población
PROB_MUT = 1        # Probabilidad de mutacion
PX = 0.75               # Probabilidad de cruza
GMAX = 30               # Cantidad máxima de generaciones que se ejecutará el algoritmo
top_features_totrack = 200 
nexperiments = 30
# params vae
best_params = {
    'hiden1': 835,
    'hiden2': 308,
    'latent_dim': 25,
    'lr': 0.00015503766948351942,
    "epochs": 1364        
}


# data
Xtrain, y_train, Xtest, y_test, features = load_and_preprocess_data(train_dir=train_dir, test_dir=test_dir,
                                                            class_column_name=class_column, 
                                                            class_value_1=class_1)

print(f'Xtrain original {Xtrain.shape}')
print(f'Xtest original {Xtest.shape}')
print(f'y_train original {y_train.shape}')
print(f'y_test original {y_test.shape}')


DAT_SIZE = Xtrain.shape[0]  
IND_SIZE = Xtrain.shape[1]  # Cantidad de genes en el cromosoma
PM = PROB_MUT / IND_SIZE    # Probabilidad de mutación [aproximadamente 1 gen por cromosoma]
                            # Experimento 4: con mayor probabilidad de mutación.
                            # PM = 20./IND_SIZE __experimento 2 mejoró el fitness y acc en 
                            # la segunda generación pero luego se estancó

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
toolbox.register("attribute", bin, p=0.01)  # Nombre con el que se registra el componente

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



for nexperiment in range(0, nexperiments):    

    # INICIALIZAMOS UN LOGGER
    logbook = tools.Logbook()

    # Corremos el modelo con todas las features
    if nexperiment == 0:
        allf = fitness(np.ones(Xtrain.shape[1]), Xtrain, Xtest, y_train, y_test)    
        print(f"All features: FITNESS: {allf[0]:.4} -- NGENES: {int(Xtrain.shape[1])} -- Acc: {allf[1]:.4}\n")    
    # ================================================
    # INICIALIZAMOS LA POBLACIÓN
    # ================================================
    pop = toolbox.population(n=POP_SIZE)  # Inicializamos una población
    # ================================================
    # EVALUAMOS EL FITNESS DE LA POBLACION
    # ======================================
    # fitnesses = list(map(toolbox.evaluate, pop))
    fitnesses = Parallel(n_jobs=16, backend="multiprocessing")(
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
        fitnesses = Parallel(n_jobs=16, backend="multiprocessing")(
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

        # file_path = f"{current_dir}/{experiment_name}.txt"
        # with open(file_path, 'a') as file:
        #     if g % 1 == 0:
        #         file.write("=" * 79 + "\n")
        #         file.write(f"GENERATION: {g}\n")
        #         file.write(
        #             f"Elite -- Fitness: {elite.fitness.values[0]:.4} -- NGENES: {np.sum(elite)} -- Acc: {elite.acc:.4}\n"
        #         )
        #         file.write("Poblacion FITNES: " + str(records["fitness"]) + "\n")
        #         file.write("Poblacion ACC: " + str(records["acc"]) + "\n")
        #         file.write("Poblacion GENES: " + str(records["ngenes"]) + "\n")
        #         file.write("#" * 79 + "\n")
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
    
    # Consolidate common features
    # assign 1 if feature was activated 1 o more times 
    common_genome_bin = [int(any(column)) for column in zip(*pop)]
    # calculate the frecuency of activated feature in population
    common_genome_prob = [sum(column)/POP_SIZE for column in zip(*pop)]
    # Sort frecuency    
    common_genome_prob_sort = sorted(common_genome_prob, reverse=True)
    # Get the ith element of the sorted list of frecuency       
    ngenome_value = common_genome_prob_sort[top_features_totrack]
    # Create a mask list to select features with grater activation frecuency    
    prob_mask = [1 if feature_include >= ngenome_value else 0 for feature_include in common_genome_prob]
    # Select the ith features with grater activation frecuency
    
    selected_features = [string for string, include in zip(features, prob_mask) if include == 1]
    selected_features_bin = [string for string, include in zip(features, common_genome_bin) if include == 1]

    print(f'common_genome {np.sum(common_genome_bin)}')
    print(f'selected_features {len(selected_features)}')

    # Define your JSON file path
    json_file_path = current_dir + '/experiments.json'
    # Check if the file exists and read its content; if not, initialize an empty dictionary
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            experiments_data = json.load(file)
    else:
        experiments_data = {}
    # Update the dictionary with new experiment data
    experiments_data[f'{experiment_name}_{nexperiment}'] = {
        "common_genome_bin": common_genome_bin,
        "common_genome_bin_sum": int(np.sum(common_genome_bin)),  # Convert to int for JSON compatibility
        "selected_features_bin": selected_features_bin, 
        "common_genome_prob_sum": len(selected_features),  # Convert to int for JSON compatibility
        "selected_features_prob": selected_features,         
        "common_genome_all_prob": common_genome_prob, 
    }
    # Write the updated dictionary back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(experiments_data, file, indent=4)
    print(f'Updated experiment data saved to {json_file_path}.')

    # Preparing data with all float numbers rounded to 3 decimals
    data = {
        'experiment_name': f'{experiment_name}_{nexperiment}',
        'date': datetime.now().date(),
        'description': description,
        'current_dir': current_dir,        
        'POP_SIZE': POP_SIZE,
        'PROB_MUT': round(PROB_MUT, 3),
        'PX': round(PX, 3),
        'GMAX': GMAX,
        'DAT_SIZE': DAT_SIZE,       

        'all_features_fitness': round(allf[0], 3), 
        'all_feature_ngenes': IND_SIZE, 
        'all_feature_acc': round(allf[1], 3),

        'elite_fitness': round(elite.fitness.values[0], 3),
        'elite_ngenes': np.sum(elite),
        'elite_acc': round(elite.acc, 3),

        'pob_fitness_avg': round(final_records['fitness']['avg'], 3),
        'pob_accuracy_avg': round(final_records['acc']['avg'], 3),
        'pob_ngenes_avg': final_records['ngenes']['avg'],
        
        'pob_fitness_std': round(final_records['fitness']['std'], 3),
        'pob_accuracy_std': round(final_records['acc']['std'], 3),
        'pob_ngenes_std': round(final_records['ngenes']['std'], 3),

        'pob_fitness_max': round(final_records['fitness']['max'], 3),
        'pob_accuracy_max': round(final_records['acc']['max'], 3),
        'pob_ngenes_max': final_records['ngenes']['max'],

    }

    # Define CSV file path
    csv_file_path = f'{current_dir}/experiments_results.csv'

    # Check if the file already exists
    file_exists = os.path.exists(csv_file_path)

    # Writing or appending data to CSV
    with open(csv_file_path, mode='a' if file_exists else 'w', newline='') as file:
        fieldnames = data.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header only if file does not exist (i.e., writing for the first time)
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)

    print(f'Results {"appended to" if file_exists else "saved in"} {csv_file_path}')

    # Generate and log plots as artifacts
    fitness_plot_path = plot_evolution(logbook, "fitness", "Fitness", 
                                       current_dir=current_dir, 
                                       experiment_name=f'{experiment_name}_{nexperiment}',
                                       filename='fitness_evolution.png', 
                                       GMAX = GMAX)
    acc_plot_path = plot_evolution(logbook, "acc", "Accuracy", 
                                   current_dir=current_dir, 
                                   experiment_name=f'{experiment_name}_{nexperiment}',
                                   filename= 'accuracy_evolution.png', 
                                   GMAX = GMAX)
    genes_plot_path = plot_evolution(logbook, "ngenes", "Number of Genes",
                                    experiment_name=f'{experiment_name}_{nexperiment}',
                                    filename='genes_evolution.png', 
                                    current_dir=current_dir, 
                                    N_override= GMAX)
