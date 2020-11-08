from book_code.Chapter04.tsp import TravelingSalesmanProblem
from deap import creator, tools, base, algorithms

import random
import array
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Problem constants
TSP_NAME = 'bayg29'
# Algorithm constants
POPULATION_SIZE = 300
MAX_GENERATIONS = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
HALL_OF_FAME_SIZE = 1
# Seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# Create TSP environment
tsp = TravelingSalesmanProblem(TSP_NAME)
# Fitness Strategy
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
# Create toolbox
toolbox = base.Toolbox()
# Individual Creator
creator.create('Individual', array.array, typecode='i', fitness=creator.FitnessMin)
toolbox.register('random_shuffle', random.sample, range(len(tsp)), len(tsp))
toolbox.register('individual_creator', tools.initIterate, creator.Individual, toolbox.random_shuffle)
# Population Creator
toolbox.register('population_creator', tools.initRepeat, list, toolbox.individual_creator, POPULATION_SIZE)
# Evaluation Function
def tsp_distance(individual):
    return (tsp.getTotalDistance(individual),) # Needs to return a tuple, hence the wrapper function
toolbox.register('evaluate', tsp_distance)
# Genetic Operators
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0/len(tsp))

def main():
    # Stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)
    # Hall of fame
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Initial population
    population = toolbox.population_creator()

    # Run genetic algorithm
    population, logbook = algorithms.eaSimple(population, toolbox,
                                                cxpb=P_CROSSOVER,
                                                mutpb=P_MUTATION,
                                                ngen=MAX_GENERATIONS,
                                                stats=stats,
                                                halloffame=hof,
                                                verbose=True)

    # print best individual info:
    best = hof.items[0]
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # plot best solution:
    plt.figure(1)
    tsp.plotData(best)

    # Plot statistics
    min_fitness_values, avg_fitness_values = logbook.select('min', 'avg')
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(min_fitness_values, color='red')
    plt.plot(avg_fitness_values, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')


    # show both plots:
    plt.show()

if __name__ == '__main__':
    main()