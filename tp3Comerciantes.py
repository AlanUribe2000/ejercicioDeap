import random
from deap import base
from deap import creator
from deap import tools
import numpy as np

distances = np.zeros((6, 6))

distances[1][0] = \
    distances[0][1] = 12
distances[2][0] = \
    distances[0][2] = 8
distances[3][0] = \
    distances[0][3] = 41
distances[4][0] = \
    distances[0][4] = 14
distances[5][0] = \
    distances[0][5] = 34

distances[2][1] = \
    distances[1][2] = 8
distances[3][1] = \
    distances[1][3] = 5
distances[4][1] = \
    distances[1][4] = 1
distances[5][1] = \
    distances[1][5] = 10

distances[3][2] = \
    distances[2][3] = 9
distances[4][2] = \
    distances[2][4] = 6
distances[5][2] = \
    distances[2][5] = 3

distances[4][3] = \
    distances[3][4] = 4
distances[5][3] = \
    distances[3][5] = 11

distances[5][4] = \
    distances[4][5] = 16

print(distances)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
## permutation setup for individual,
toolbox.register("indices", \
                 random.sample, \
                 range(6), 
                 6)
toolbox.register("individual", \
                 tools.initIterate, \
                 creator.Individual, toolbox.indices)
## population setup,
toolbox.register("population", \
                 tools.initRepeat, \
                 list, toolbox.individual)

# [ individual[:10] for individual in toolbox.population(n=5) ]

def EVALUATE(individual):
    summation = 0
    start = individual[0]
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances[start][end]
        start = end
    summation += distances[0][5]
    return summation
toolbox.register("evaluate", EVALUATE)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=10)

POPULATION_SIZE = 200
N_ITERATIONS = 1000
N_MATINGS = 50
a = Runner(toolbox)
a.set_parameters(POPULATION_SIZE, N_ITERATIONS, N_MATINGS)
stats, population = a.Run()