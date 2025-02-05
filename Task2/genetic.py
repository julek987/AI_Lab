import random
import pandas as pd
import numpy as np

# Step 1: Data Preparation
def read_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Arkusz2', header=None)
    tasks = []
    for i in range(0, len(df.columns), 2):
        resource = df.iloc[2, i]
        time = df.iloc[2, i+1]
        tasks.append((resource, time))
    return tasks

# Step 2: Chromosome Representation
def create_chromosome(tasks):
    return random.sample(range(len(tasks)), len(tasks))

# Step 3: Initial Population
def create_population(pop_size, tasks):
    return [create_chromosome(tasks) for _ in range(pop_size)]

# Step 4: Fitness Function (Modified)
def calculate_fitness(chromosome, tasks):
    resource_time = {}
    for task_index in chromosome:
        resource, time = tasks[task_index]
        if resource in resource_time:
            resource_time[resource] += time
        else:
            resource_time[resource] = time
    # Return the difference between max and min resource time to diversify
    return max(resource_time.values()) - min(resource_time.values())

# Step 5: Tournament Selection (Modified)
def select_parents(population, fitnesses, tournament_size=5):
    selected_parents = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]  # Choose the one with the best fitness
        selected_parents.append(winner)
    return selected_parents

# Step 6: Uniform Crossover (Modified)
def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

# Step 7: Mutation (Modified)
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome)-1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

# Step 8: Evolution
def genetic_algorithm(tasks, pop_size, generations, mutation_rate):
    population = create_population(pop_size, tasks)
    best_fitness = float('inf')
    best_chromosome = None

    for generation in range(generations):
        fitnesses = [calculate_fitness(chromosome, tasks) for chromosome in population]

        # Track the best chromosome and fitness
        gen_min_fitness = min(fitnesses)
        if gen_min_fitness < best_fitness:
            best_fitness = gen_min_fitness
            best_chromosome = population[np.argmin(fitnesses)]

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))

        population = new_population

    return best_chromosome, best_fitness

# Main Execution
file_path = 'GA_task.xlsx'
tasks = read_data(file_path)
pop_size = 50
generations = 10000
mutation_rate = 0.01  # Increased mutation rate

best_chromosome, best_fitness = genetic_algorithm(tasks, pop_size, generations, mutation_rate)
print("Best Chromosome:", best_chromosome)
print("Best Fitness (Minimal Total Time Difference):", best_fitness)
