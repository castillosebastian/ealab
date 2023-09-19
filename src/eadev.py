import math
import random
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import tempfile

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

from deap import base  # Estructura que permite agrupar todos los componentes de nuestro algoritmo en una misma bolsa
from deap import creator  # Permite crear los componentes de nuestro algoritmo
from deap import tools  # Contiene funciones precargadas

from joblib import Parallel, delayed
from utils import bin, mutation, load_params, fitness

import mlflow

def main(exp_name):
    # Load parameters
    params = load_params(f'exp/{exp_name}/{exp_name}.yml')

    # Set experiment name to be the source file name
    script_name = os.path.basename(__file__)  # gets the name of the current file
    experiment_name, _ = os.path.splitext(script_name)  # removes the extension
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=exp_name):
    
        # Now you can access your parameters as follows:
        data_tr = params['data_tr']
        data_ts = params['data_ts']
        poblacion_size = params['poblacion_size']
        pm_divsize = params['pmutation']
        pc = params['pcrossover']
        gmax = params['maxgenereation']
        description = params['description']  # get the description from the YAML file        

    # Log parameters
        mlflow.log_param("data_tr", data_tr)
        mlflow.log_param("data_ts", data_ts)
        mlflow.log_param("poblacion_size", poblacion_size)
        mlflow.log_param("pm_divsize", pm_divsize)
        mlflow.log_param("pc", pc)
        mlflow.log_param("gmax", gmax)         
        mlflow.log_param("description", description)

        # Script--------------------------------------------------------------
        # load and clean data
        TRAIN = pd.read_csv(data_tr, header=None)
        TEST = pd.read_csv(data_ts, header=None)
        scaler = StandardScaler()
        TRAIN = TRAIN.to_numpy()
        TEST = TEST.to_numpy()
        X_TRAIN = TRAIN[:,:-1]
        #y_train = (TRAIN[:,-1] + 1) /2
        y_train = TRAIN[:,-1]
        y_train = np.where(np.array(y_train) == 2, 1, 0).astype('int64')
        X_TEST = TEST[:,:-1]
        y_test = TEST[:,-1]
        y_test = np.where(np.array(y_test) == 2, 1, 0).astype('int64')
        scaler.fit(X_TRAIN)
        Xtrain = scaler.transform(X_TRAIN)
        Xtest = scaler.transform(X_TEST)


        # Cromosoma -----------------------------------------------------------------------------------
        IND_SIZE = Xtrain.shape[1]  # Cantidad de genes en el cromosoma
        POP_SIZE = poblacion_size # Cantidad de individuos en la población
        PM = pm_divsize/IND_SIZE  # Probabilidad de mutación [aproximadamente 1 gen por cromosoma]
        PX = pc  # Probabilidad de cruza
        GMAX = gmax  # Cantidad máxima de generaciones que se ejecutará el algoritmo

        
        # CREAMOS LA FUNCION DE FITNESS
        # Esta función tiene "1 OBJETIVO" a "MAXIMIZAR"
        creator.create("Fitness",  # Nombre con el que se registra el componente
                    base.Fitness,  # Clase de la que hereda
                    weights=(1.0,)) 

        # CREAMOS EL CONSTRUCTOR DE INDIVIDUOS
        creator.create("Individual", # Nombre con el que se registra el componente
                    list,  # Clase de la que hereda [Tipo de contenedor en este caso]
                    fitness=creator.Fitness,
                    acc=0,
                    ngenes=0)  # Asignamos un método para evaluar el fitness del individuo

        ### REGISTRAMOS COMPONENTES

        toolbox = base.Toolbox()
        #---------------------
        # DEFINIMOS COMO CONSTRUIR UN GEN
        # el algoritmo retenie la historia de fitnes de genes activos, contribuyendo !!!IMPORTANTE
        # a la selección de las variables que contribuyen a mejorar el fitness
        toolbox.register("attribute",  # Nombre con el que se registra el componente
                        bin,
                        p=0.1)
                                
        # DEFINIMOS COMO CONSTRUIR UN INDIVIDUO/CROMOSOMA
        toolbox.register("individual",  # Nombre con el que se registra el componente
                        tools.initRepeat,  # Método usado para construir el cromosoma
                        creator.Individual,  # ...
                        toolbox.attribute,  # Función para construir cada gen
                        n=IND_SIZE)  # Número de genes del cromosoma/individuo (se repetirá la función construir gen)

        # DEFINIMOS COMO CONSTRUIR LA POBLACION
        toolbox.register("population",  # Nombre con el que se registra el componente
                        tools.initRepeat,  # Método usado para construir el cromosoma
                        list,
                        toolbox.individual)

        # DEFINIMOS COMO REALIZAR LA CRUZA
        toolbox.register("mate",  # Nombre con el que se registra el componente
                        tools.cxTwoPoint)  # 


        # DEFINIMOS COMO REALIZAR LA MUTACION
        toolbox.register("mutate",  # Nombre con el que se registra el componente
                        mutation,  # Método de mutación (definido como función más arriba)
                        p=PM)  # Parámetro que usa la mutación

        #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        #toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

        # DEFINIMOS COMO REALIZAR LA SELECCION DE INDIVIDUOS
        toolbox.register("select",  # Nombre con el que se registra el componente
                        tools.selTournament,  # Método usado para selección [selRoulette | selTournament | ...]
                        tournsize=5)  # Parámetro que usa el torneo

        #------------------------------------------------------
        # Stats metrics

        # EXTRAEMOS EL FITNESS DE TODOS LOS INDIVIDUOS
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])

        # EXTRAEMOS EL ACC DE TODOS LOS INDIVIDUOS
        stats_acc = tools.Statistics(key=lambda ind: ind.acc)

        # EXTRAEMOS LA FRACCION DE GENES ACTIVOS DE TODOS LOS INDIVIDUOS
        stats_frac_active_genes = tools.Statistics(key=lambda ind: ind.ngenes)

        # EXTRAEMOS EL NUMERO DE GENES ACTIVOS DE TODOS LOS INDIVIDUOS
        stats_active_genes = tools.Statistics(key=lambda ind: np.sum(ind))

        mstats = tools.MultiStatistics(fitness=stats_fit,
                                    acc=stats_acc,
                                    frac_ngenes=stats_frac_active_genes,
                                    ngenes=stats_active_genes)

        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        # INICIALIZAMOS UN LOGGER
        logbook = tools.Logbook()
        #================================================
        # INICIALIZAMOS LA POBLACIÓN
        #================================================
        pop = toolbox.population(n=POP_SIZE)  # Inicializamos una población
        #================================================
        # EVALUAMOS EL FITNESS DE LA POBLACION
        #======================================
        print('Evaluamos fitness de la población')

        #fitnesses = list(map(toolbox.evaluate, pop))
        fitnesses = Parallel(n_jobs=4, backend='multiprocessing')(delayed(fitness)(ind, Xtrain, Xtest, y_train, y_test) for ind in pop)
        #================================================
        # ASIGNAMOS A CADA INDIVIDUO SU FITNESS
        #========================================
        for ind,fit in zip(pop, fitnesses):
            ind.fitness.values = (fit[0],)  # Guardamos el fitness para cada individuo (en el individuo)
            ind.acc = fit[1]
            ind.ngenes = fit[2]
        #================================================
        records = mstats.compile(pop)
        logbook.record(gen=0, **records)

        #################################################################################
        # COMENZAMOS LA EVOLUCION
        ################################
        print("Comenzamos la evolución")

        for g in range(1,GMAX):#tqdm(range(GMAX)):

            #================================================
            # SELECCIONAMOS INDIVIDUO ELITE
            #================================
            idx_elite = np.argmax(fitnesses, axis=0)[0]  # Si maximizamos, hay que usar ".argmax()". El indiv. con mejor fitness
            elite = toolbox.clone(pop[idx_elite])
            del elite.fitness.values, elite.acc, elite.ngenes
            #================================================

            
            #================================================
            # HACEMOS UNA COPIA DE LA POBLACION ACTUAL
            #==========================================
            parents = toolbox.select(pop, POP_SIZE)  # Seleccionamos individuos para alcanzar
                                                    # el tamaño de la población
            
            offspring = list(map(toolbox.clone, pop))  # Clonamos para tener nuevos individuos
                                                            # (contenedores independientes)
            #============================================
            
            # REALIZAMOS LA CRUZA DE LOS PADRES
            #====================================
            for i in range(POP_SIZE//2):
                parent1 = toolbox.clone(parents[random.randint(0,POP_SIZE-1)])
                parent2 = toolbox.clone(parents[random.randint(0,POP_SIZE-1)])
                
                if random.random() < PX:
                    childs = toolbox.mate(parent1, parent2)
                else:
                    childs = (parent1, parent2)
                
                offspring[2*i] = childs[0]
                offspring[2*i+1] = childs[1]


            #================================================
            # MUTAMOS LOS HIJOS
            #=================================
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values  #, mutant.acc, mutant.ngenes
            #================================================
            
            
            #================================================
            # EVALUAMOS EL FITNESS
            # Y SE LO ASIGNAMOS A CADA INDIVIDUO
            #======================================
            offspring[0] = elite
            
            fitnesses = Parallel(n_jobs=4, backend='multiprocessing')(delayed(fitness)(ind, Xtrain, Xtest, y_train, y_test) for ind in offspring)

            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = (fit[0],)  # Guardamos el fitness para cada individuo (en el individuo)
                ind.acc = fit[1]
                ind.ngenes = fit[2]
            #================================================
            
            #================================================
            # CONSTRUIMOS LA NUEVA POBLACION
            #================================
            pop = toolbox.clone(offspring)
            #================================================
            
            #================================================
            # CALCULAMOS ESTADÏSTICAS
            #============================
            records = mstats.compile(pop)
            logbook.record(gen=g, **records)

            # Log metrics for this generation            
            mlflow.log_metric("evol_elite_fitness", elite.fitness.values[0], step=g)
            mlflow.log_metric("evol_elite_acc", elite.acc, step=g)
            mlflow.log_metric("evol_elite_ngenes", np.sum(elite), step=g)            
            mlflow.log_metric("evol_pop_fitness_avg", records['fitness']['avg'], step=g)
            mlflow.log_metric("evol_pop_acc_avg", records['acc']['avg'], step=g)
            mlflow.log_metric("evol_pop_genes_avg", records['ngenes']['avg'], step=g) 
            # Check std
            mlflow.log_metric("evol_pop_fitness_std", records['fitness']['std'], step=g)
            mlflow.log_metric("evol_pop_acc_std", records['acc']['std'], step=g)
            mlflow.log_metric("evol_pop_genes_std", records['ngenes']['std'], step=g)        
            
            if (g%1 == 0):
                print('='*79)
                print(f'GENERATION: {g}')
                print(f'ELITE -- Fitness: {elite.fitness.values[0]:.4} -- NGENES: {np.sum(elite)} -- Acc: {elite.acc:.4}')
                print('FITNES: ', records['fitness'])
                print('ACC: ', records['acc'])
                print('GENES: ', records['ngenes'])

        # Log metrics for all the evolution        
        mlflow.log_metric("final_max_fitness", records['fitness']['max'])
        mlflow.log_metric("final_avg_fitness", records['fitness']['avg'])
        mlflow.log_metric("final_max_acc", records['acc']['max'])
        mlflow.log_metric("final_avg_acc", records['acc']['avg'])
        mlflow.log_metric("final_genes", records['ngenes']['max'])        

        f_avg = logbook.chapters['fitness'].select('avg')  # Extraemos fitness promedio a lo largo de las épocas
        f_max = logbook.chapters['fitness'].select('max')  # Extraemos fitness máximo a lo largo de las épocas
        f_min = logbook.chapters['fitness'].select('min')  # Extraemos fitness mínimo a lo largo de las épocas

        N = 30 if GMAX > 200 else GMAX 
        fig, ax = plt.subplots(1, 1, figsize=(20,6)) 
        ax.plot(range(N), f_avg[:N], '-or')
        ax.plot(range(N), f_max[:N], '-og')
        ax.plot(range(N), f_min[:N], '-ob')
        ax.set_xlabel('Generaciones', fontsize=16)
        ax.set_ylabel('Fitness', fontsize=16)
        ax.grid(True)

        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            fig.savefig(temp.name)
            # Log the figure as an artifact in MLflow
            mlflow.log_artifact(temp.name, "fitness_plots.png")

        f_avg = logbook.chapters['acc'].select('avg')  # Extraemos fitness promedio a lo largo de las épocas
        f_max = logbook.chapters['acc'].select('max')  # Extraemos fitness máximo a lo largo de las épocas
        f_min = logbook.chapters['acc'].select('min')  # Extraemos fitness mínimo (elite) a lo largo de las épocas

        N = 30 if GMAX > 200 else GMAX 
        fig, ax = plt.subplots(1, 1, figsize=(20,6)) 
        ax.plot(range(N), f_avg[:N], '-or')
        ax.plot(range(N), f_max[:N], '-og')
        ax.plot(range(N), f_min[:N], '-ob')
        ax.set_xlabel('Generaciones', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.grid(True)

        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            fig.savefig(temp.name)
            # Log the figure as an artifact in MLflow
            mlflow.log_artifact(temp.name, "accuracy_plots.png")

        ngenes_avg = logbook.chapters['ngenes'].select('avg')  # Extraemos fitness promedio a lo largo de las épocas
        ngenes_max = logbook.chapters['ngenes'].select('max')  # Extraemos fitness máximo a lo largo de las épocas
        ngenes_min = logbook.chapters['ngenes'].select('min')  # Extraemos fitness mínimo (elite) a lo largo de las épocas

        N = 100 if GMAX > 2000 else GMAX 
        fig, ax = plt.subplots(1, 1, figsize=(20,6)) 
        ax.plot(range(N), ngenes_avg[:N], '-or')
        ax.plot(range(N), ngenes_max[:N], '-og')
        ax.plot(range(N), ngenes_min[:N], '-ob')
        ax.set_xlabel('Generaciones', fontsize=16)
        ax.set_ylabel('Número de genes', fontsize=16)
        ax.grid(True)   

        with tempfile.NamedTemporaryFile(suffix=".png") as temp:
            fig.savefig(temp.name)
            # Log the figure as an artifact in MLflow
            mlflow.log_artifact(temp.name, "genes_plots.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with given name')
    parser.add_argument('exp_name', type=str, help='The name of the experiment to run')
    args = parser.parse_args()

    main(args.exp_name)