from deap import tools, base, creator

import random
import matplotlib.pyplot as plt
import seaborn as sns

# Problem constants
ONE_MAX_LENGTH = 100
# Genetic Algorithm constants
POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
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
    generation_count = 0
    population = toolbox.population_creator(n=POPULATION_SIZE)
    # Evaluate initial population
    fitness_values = list(map(toolbox.evaluate, population))
    for individual, fitness_value in zip(population, fitness_values):
        individual.fitness.values = fitness_value

    # Statstics collection
    fitness_values = [individual.fitness.values[0] for individual in population]
    max_fitness_values = []
    mean_fitness_values = []

    while generation_count < MAX_GENERATIONS and max(fitness_values) < ONE_MAX_LENGTH:
        generation_count += 1
        # Apply genetic operators
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring)) # clone the offspring
        # Mate
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2) # in-place operation
                # Clear the old fitness values
                del child1.fitness.values
                del child2.fitness.values
        # Mutate
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values # Needs to re-calculated
        # Calculate fitness for new individuals
        fresh_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fresh_fitness_values = list(map(toolbox.evaluate, fresh_individuals))
        for fitness_value, individual in zip(fresh_fitness_values, fresh_individuals):
            individual.fitness.values = fitness_value
        # Replace old population
        population = offspring

        # Statistics
        fitness_values = [ind.fitness.values[0] for ind in population]
        max_fitness = max(fitness_values)
        mean_fitness = sum(fitness_values)/len(fitness_values)
        best_index = fitness_values.index(max_fitness)
        max_fitness_values.append(max_fitness)
        mean_fitness_values.append(mean_fitness)
        print(f'Generation {generation_count}: max_fitness = {max_fitness:.2f}, mean_fitness = {mean_fitness:.2f}')
        #print(f'Best individual = {population[best_index]}')

    # Genetic Algorithm is done - plot statistics:
    sns.set_style("whitegrid")
    plt.plot(max_fitness_values, color='red')
    plt.plot(mean_fitness_values, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max/Avg Fitness')
    plt.title('Max and Avg fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()