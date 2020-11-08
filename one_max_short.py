from deap import tools, base, creator, algorithms

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Problem constants
ONE_MAX_LENGTH = 100
# Genetic Algorithm constants
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5
# SEED
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create toolbox
toolbox = base.Toolbox()
# Create function to randomly create 0 or 1
toolbox.register('zero_or_one', random.randint, 0, 1)
# Single objective, maximizing strategy
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
# Create the individual class, based on list
creator.create('Individual', list, fitness=creator.FitnessMax)
# Create the individual creator operator
toolbox.register('individual_creator', tools.initRepeat,
                creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGTH)
# Create population creator to generate a list of individuals
toolbox.register('population_creator', tools.initRepeat,
                list, toolbox.individual_creator) # Population size not given here, should be specified when calling
# Fitness evaluation
def one_max_fitness(individual):
    return (sum(individual),) # Return a tuple, since DEAP expects a tuple for fitness values

toolbox.register('evaluate', one_max_fitness)
# Genetic operators
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', tools.cxOnePoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


def main():
    # Initial population
    population = toolbox.population_creator(n=POPULATION_SIZE)
    # Evaluate initial population
    fitness_values = list(map(toolbox.evaluate, population))
    for individual, fitness_value in zip(population, fitness_values):
        individual.fitness.values = fitness_value

    # Statstics collection
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)

    # Hall of Fame
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # Run algorithm
    population, logbook = algorithms.eaSimple(population, toolbox,
                                            cxpb=P_CROSSOVER,
                                            mutpb=P_MUTATION,
                                            ngen=MAX_GENERATIONS,
                                            stats=stats,
                                            halloffame=hof,
                                            verbose=True)
    # Extract statistics
    max_fitness_values, mean_fitness_values = logbook.select('max', 'avg')


    # Print hall of fame
    print("Hall of Fame individuals: ", *hof.items, sep='\n')
    # Plot statistics
    sns.set_style("whitegrid")
    plt.plot(max_fitness_values, color='red')
    plt.plot(mean_fitness_values, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max/Avg Fitness')
    plt.title('Max and Avg fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()