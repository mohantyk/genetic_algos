from deap import base, creator, tools
from book_code.Chapter04.elitism import eaSimpleWithElitism

import random
import numpy as np

def eggholder(individual):
    x = individual[0]
    y = individual[1]
    f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
    return f,  # return a tuple

# RANDOM SEED
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# PROBLEM CONSTANTS
BOUND_LOW = -512
BOUND_HIGH = 512
DIMENSIONS = 2

# GENETIC ALGO CONSTANTS
POPULATION_SIZE = 100
NUM_GENERATIONS = 200
TOURNAMENT_SIZE = 2
P_CROSSOVER = 0.9
P_MUTATION = 0.1
CROWDING_FACTOR = 20
HALL_OF_FAME_SIZE = 20


# Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)
# God
toolbox = base.Toolbox()
toolbox.register('random_float', random.uniform, BOUND_LOW, BOUND_HIGH)
toolbox.register('individual_creator', tools.initRepeat, creator.Individual, toolbox.random_float, DIMENSIONS)
toolbox.register('population_creator', tools.initRepeat, list, toolbox.individual_creator)

# Evaluate
toolbox.register('evaluate', eggholder)

# Genetic operators
toolbox.register('select', tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_HIGH, eta=CROWDING_FACTOR)
toolbox.register('mutate', tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_HIGH, eta=CROWDING_FACTOR, indpb=1.0/DIMENSIONS)

def main():
    population = toolbox.population_creator(n=POPULATION_SIZE)
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register('min', np.min)
    stats.register('avg', np.mean)

    # Hall of Fame
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    population, logbook = eaSimpleWithElitism(population, toolbox,
                                            cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                            ngen=NUM_GENERATIONS,
                                            stats=stats, halloffame=hof,
                                            verbose=True)

    # Print results
    best = hof.items[0]
    best_value = best.fitness.values[0]
    print(f'Best value = {best_value:.3f} @ {best}')


if __name__ == '__main__':
    main()