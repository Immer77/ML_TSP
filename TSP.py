"""
Created on Wed Mar 3 16:39:18 2021
@author: Sila
"""

# Gettting started methods for TSP GA algorithm
# - Read cities from file
#

import pandas as pd
import random
import math

data = pd.read_csv('TSPcities1000.txt', sep='\s+', header=None)
data = pd.DataFrame(data)

import matplotlib.pyplot as plt

x = data[1]
y = data[2]
plt.plot(x, y, 'r.')
plt.show()


def createRandomRoute():
    tour = [[i] for i in range(10)]
    random.shuffle(tour)
    return tour


# plot the tour - Adjust range 0..len, if you want to plot only a part of the tour.
def plotCityRoute(route):
    for i in range(0, len(route)):
        plt.plot(x[i:i + 2], y[i:i + 2], 'ro-')
    plt.show()


# Alternativ kode:
#  for i in range(0, len(route)-1):              
#     plt.plot([x[route[i]],x[route[i+1]]], [y[route[i]],y[route[i+1]]], 'ro-')

tour = createRandomRoute()
print(tour)
plotCityRoute(tour)


# calculate distance between cities
def distancebetweenCities(city1x, city1y, city2x, city2y):
    xDistance = abs(city1x - city2x)
    yDistance = abs(city1y - city2y)
    distance = math.sqrt((xDistance * xDistance) + (yDistance * yDistance))
    return distance


# distance between city number 100 and city number 105
dist = distancebetweenCities(x[100], y[100], x[105], y[105])
print('Distance, % target: ', dist)

best_score_progress = []  # Tracks progress

# replace with your own calculations

fitness_gen0 = 1000  # replace with your value
print('Starting best score, % target: ', fitness_gen0)

best_score = fitness_gen0
# Add starting best score to progress tracker
best_score_progress.append(best_score)


# Here comes your GA program...
def fitness(individual):
    total_distance = 0
    # For each individual route
    for i in range(len(individual) - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        total_distance += distancebetweenCities(x[city1], y[city1], x[city2], y[city2])
    # Add distance from the last city back to the starting city to get the total distance
    total_distance += distancebetweenCities(x[individual[-1]], y[individual[-1]], x[individual[0]], y[individual[0]])
    return total_distance


def crossover(parent1: str, parent2: str) -> str:
    """
    Perform crossover between two parents to create a child.
    Args:
        parent1 (str): The first parent's chromosome.
        parent2 (str): The second parent's chromosome.
    Returns:
        str: The chromosome of the child.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def selection(population: list, num_parents: int) -> list:
    """
    Select the top `num_parents` individuals from the population based on their fitness.
    Args:
        population (list): List of chromosomes (individuals) in the population.
        num_parents (int): Number of parents to select.
    Returns:
        list: List of selected parents.
    """
    # Calculate the fitness of each individual in the population
    fitness_scores = [fitness(individual) for individual in population]

    # Sort the population by fitness in descending order
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]

    # Select the top `num_parents` individuals as parents
    selected_parents = sorted_population[:num_parents]

    return selected_parents


def mutate(route_to_mut):
    '''
	Route() --> Route()
	Swaps two random indexes in route_to_mut.route. Runs
	k_mut_prob*100 % of the time
	'''
    # k_mut_prob %
    if random.random() < route_to_mut:
        # two random indices:
        mut_pos1 = random.randint(0, len(route_to_mut.route) - 1)
        mut_pos2 = random.randint(0, len(route_to_mut.route) - 1)
        # if they're the same, skip to the chase
        if mut_pos1 == mut_pos2:
            return route_to_mut
        # Otherwise swap them:
        city1 = route_to_mut.route[mut_pos1]
        city2 = route_to_mut.route[mut_pos2]
        route_to_mut.route[mut_pos2] = city1
        route_to_mut.route[mut_pos1] = city2


best_score = 980
best_score_progress.append(best_score)
best_score = 960
best_score_progress.append(best_score)



# GA has completed required generation
print('End best score, % target: ', best_score)

plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best Fitness - route length - in Generation')
plt.show()

# Define your population
population_size = 1000
population = [createRandomRoute() for _ in range(population_size)]

# Main GA loop
for generation in range(num_generations):
    # Calculate fitness for each individual
    for individual in population:
        individual['fitness'] = fitness(individual)

    # Select parents for crossover
    selected_parents = tournament_selection(population, tournament_size)

    # Create a new population through crossover and mutation
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_parents, 2)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            mutate(child)
        new_population.append(child)

    # Replace the old population with the new population
    population = new_population

    # Track the best fitness in this generation
    best_fitness = min(individual['fitness'] for individual in population)
    best_score_progress.append(best_fitness)

# Print or visualize the best route found
best_route = min(population, key=lambda x: x['fitness'])
print('Best route:', best_route)
plotCityRoute(best_route)
