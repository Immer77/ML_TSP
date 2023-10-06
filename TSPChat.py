import time

import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# Read city data from a file
data = pd.read_csv('TSPcities1000.txt', sep='\s+', header=None)

# Extract city coordinates
x = data[1]
y = data[2]
plt.plot(x, y, 'r.')
plt.show()

"""
Creating random routes
"""
def createRandomRoute(num_cities):
    # Create a random route by shuffling city indices
    tour = random.sample(range(num_cities), num_cities)
    return tour

"""
Function to plot city routes
"""
def plotCityRoute(route, x, y):
    # Plot the TSP route
    for i in range(len(route) - 1):
        plt.plot([x[route[i]], x[route[i + 1]]], [y[route[i]], y[route[i + 1]]], 'ro-')
    plt.show()

"""
Function to calculate the distance between two cities
"""
def distanceBetweenCities(city1x: int, city1y: int, city2x: int, city2y: int) -> int:
    # Calculate the Euclidean distance between two cities
    xDistance = abs(city1x - city2x)
    yDistance = abs(city1y - city2y)
    distance = math.sqrt(xDistance ** 2 + yDistance ** 2)
    return distance

"""
Fitness function to calculate the total distance of a route
used to evaluate the fitness of a route (lower distance is better)
"""
def fitness(route):
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        total_distance += distanceBetweenCities(x[city1], y[city1], x[city2], y[city2])
    total_distance += distanceBetweenCities(x[route[-1]], y[route[-1]], x[route[0]], y[route[0]])
    return total_distance

"""
function to select parents for crossover
The selection process is based on the fitness of each individual in the population
It is then sorted in ascending order (lowest fitness first)
and finally the lowest num_parents individuals are selected as parents
"""
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

    # Sort the population by fitness in ascending order (lowest fitness first)
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]

    # Select the top `num_parents` individuals as parents
    selected_parents = sorted_population[:num_parents]

    return selected_parents

"""
Crossover function to create a child from two parents

"""
def crossover(parent1, parent2):
    # Perform order-based crossover between two parents
    # Choose a random segment from the first parent
    start_idx = random.randint(0, len(parent1) - 1)
    # The end index must be greater than the start index
    end_idx = random.randint(start_idx + 1, len(parent1))
    # The selected segment from the first parent
    segment = parent1[start_idx:end_idx]
    # The remaining cities from the second parent
    child = [city for city in parent2 if city not in segment]
    # The new child
    child[start_idx:start_idx] = segment
    return child

"""
Mutation function to mutate a route
Mutation is performed by swapping two random cities in the route
Might be a problem if it is a city he has already visited
"""
def mutate(route) -> None:
    # Swap two random cities in the route
    if random.random() < mutation_rate:
        mut_pos1 = random.randint(0, len(route) - 1)
        mut_pos2 = random.randint(0, len(route) - 1)
        route[mut_pos1], route[mut_pos2] = route[mut_pos2], route[mut_pos1]

start = time.time()
# Configuration
num_cities = 100
population_size = 100
num_generations = 1000
mutation_rate = 0.2

# Initialize the population with random routes
population = [createRandomRoute(num_cities) for _ in range(population_size)]

# Track the best fitness scores
best_score_progress = []

# Main GA loop
for generation in range(num_generations):
    # Calculate fitness for each individual
    fitness_scores = [fitness(individual) for individual in population]

    # Select parents for crossover
    selected_parents = selection(population, population_size // 2)

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
    best_fitness = min(fitness_scores)
    best_score_progress.append(best_fitness)

# Print or visualize the best route found
best_route_idx = fitness_scores.index(min(fitness_scores))
best_route = population[best_route_idx]
print('Best route:', best_route)
print('Best Route distance:', min(best_score_progress))
plotCityRoute(best_route, x, y)
end = time.time()
print(f"this took : {end - start} seconds")
# Plot the progress of the best fitness scores
plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best Fitness - Total Distance')
plt.show()
