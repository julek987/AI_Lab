import pandas as pd
import random

# Read the Excel file
df = pd.read_excel('GA_task.xlsx', header=None)

# Initialize a list to hold all the orders
orders = []

# Iterate over columns in pairs (R-T, R-T, etc.)
for col in range(0, df.shape[1], 2):  # Step by 2 to get pairs of columns
    order = []

    # Iterate through each row
    for i in range(2, df.shape[0]):  # Start from the 3rd row (index 2)
        # Ensure non-null values
        if pd.notna(df.iloc[i, col]) and pd.notna(df.iloc[i, col + 1]):
            order.append((df.iloc[i, col], df.iloc[i, col + 1]))

    # Append the order to the orders list
    orders.append(order)

# Print the result
print(orders)
print(len(orders))


def create_chromosome(orders):
    """
    Create a chromosome representing a possible interleaving of operations.

    Each order is represented by an integer (its index in orders). Since the
    order of operations in an order is fixed, we only need to decide how the operations
    from different orders interleave. The chromosome is a list containing each order
    repeated as many times as it has operations.
    """
    chromosome = []
    for order_id, order in enumerate(orders):
        chromosome += [order_id] * len(order)
    random.shuffle(chromosome)
    return chromosome


# Example usage:
chromosome = create_chromosome(orders)
print("Chromosome:", chromosome)


def decode_chromosome(chromosome, orders):
    """
    Decode a chromosome into a full schedule.

    Returns:
      schedule: a list of scheduled operations, each represented as a tuple:
                (order_id, operation_index, machine, start_time, finish_time)
      makespan: the total time to finish all operations (i.e. the maximum finish time)
    """
    # For each machine, track when it becomes available.
    machine_available = {}
    # For each order, track when the previous operation finished.
    order_ready_time = [0] * len(orders)
    # For each order, track which operation is next (initially 0 for all orders).
    next_op_index = [0] * len(orders)

    schedule = []  # To store the scheduled operations

    # Process the chromosome from left to right.
    for order_id in chromosome:
        op_index = next_op_index[order_id]
        # Check if all operations for this order are already scheduled.
        if op_index >= len(orders[order_id]):
            continue  # This gene is extra; it might happen if a chromosome isn’t properly repaired.

        machine, duration = orders[order_id][op_index]
        # The operation can start only when both the order and the machine are ready.
        start_time = max(order_ready_time[order_id], machine_available.get(machine, 0))
        finish_time = start_time + duration

        # Record the scheduled operation.
        schedule.append((order_id, op_index, machine, start_time, finish_time))

        # Update the ready times.
        order_ready_time[order_id] = finish_time
        machine_available[machine] = finish_time

        # Move to the next operation in the order.
        next_op_index[order_id] += 1

    makespan = max(order_ready_time)
    return schedule, makespan


# Decode the chromosome to get the schedule and makespan.
schedule, makespan = decode_chromosome(chromosome, orders)
print("Schedule:")
for op in schedule:
    print(f"Order {op[0]} Operation {op[1]} on Machine {op[2]}: start at {op[3]}, finish at {op[4]}")
print("Makespan:", makespan)


### 3. Genetic Algorithm Components

# 3.1 Create Initial Population
def create_population(orders, pop_size):
    """
    Generate an initial population of chromosomes.
    """
    return [create_chromosome(orders) for _ in range(pop_size)]


# 3.2 Evaluate Fitness
def evaluate_population(population, orders):
    """
    Evaluate each chromosome in the population.
    Returns a list of fitness values (here, the makespan).
    Lower makespan means a better schedule.
    """
    fitnesses = []
    for chromosome in population:
        _, makespan = decode_chromosome(chromosome, orders)
        fitnesses.append(makespan)
    return fitnesses


# 3.3 Tournament Selection
def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Select one parent by tournament selection.
    """
    participants = random.sample(list(zip(population, fitnesses)), tournament_size)
    # Lower fitness (makespan) is better.
    best = min(participants, key=lambda x: x[1])
    return best[0]


# 3.4 Crossover Operator
def crossover(parent1, parent2):
    """
    Custom crossover for permutations with repetition.

    For each job, we record the positions (indices) where that job appears in each parent.
    Then, for each occurrence of the job we compute the average index (from parent1 and parent2).
    Sorting all job-occurrence pairs by these average indices produces the child.

    This operator preserves the number of times each job appears.
    """
    length = len(parent1)
    jobs = set(parent1)  # All job ids
    # Build dictionaries mapping job -> list of positions in each parent.
    positions1 = {job: [] for job in jobs}
    positions2 = {job: [] for job in jobs}

    for i, gene in enumerate(parent1):
        positions1[gene].append(i)
    for i, gene in enumerate(parent2):
        positions2[gene].append(i)

    candidate_positions = []
    for job in jobs:
        occ1 = positions1[job]
        occ2 = positions2[job]
        occ1.sort()
        occ2.sort()
        # For each occurrence, compute the average of positions.
        for i in range(len(occ1)):
            avg = (occ1[i] + occ2[i]) / 2.0
            candidate_positions.append((avg, job))

    # Sort by the computed average positions.
    candidate_positions.sort(key=lambda x: x[0])
    # The child is the sequence of job ids in this sorted order.
    child = [job for _, job in candidate_positions]
    return child


# 3.5 Mutation Operator
def mutate(chromosome, mutation_rate=0.1):
    """
    Simple swap mutation.
    For each gene, with probability mutation_rate, swap it with another randomly chosen gene.
    This preserves the overall job counts.
    """
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


### 4. Genetic Algorithm Main Loop

def genetic_algorithm(orders,
                      pop_size=50,
                      generations=100,
                      crossover_rate=0.8,
                      mutation_rate=0.1,
                      tournament_size=3):
    """
    Run the genetic algorithm.

    Parameters:
      - orders: list of orders (each a list of operations).
      - pop_size: number of individuals in the population.
      - generations: number of generations to run.
      - crossover_rate: probability of performing crossover.
      - mutation_rate: probability of mutating a gene.
      - tournament_size: number of individuals in tournament selection.

    Returns:
      best_chromosome: the best solution found.
      best_fitness: its makespan.
    """
    population = create_population(orders, pop_size)
    best_chromosome = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitnesses = evaluate_population(population, orders)

        # Update best solution found so far.
        for i, fit in enumerate(fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_chromosome = population[i]

        new_population = []
        while len(new_population) < pop_size:
            # Select two parents using tournament selection.
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            # Crossover.
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation.
            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population
        print(f"Generation {gen + 1}: Best Makespan = {best_fitness}")

    return best_chromosome, best_fitness


# Run the genetic algorithm.
best_chrom, best_fit = genetic_algorithm(orders,
                                         pop_size=1500,
                                         generations=100,
                                         crossover_rate=0.6,
                                         mutation_rate=0.01,
                                         tournament_size=40)

print("\nBest Found Schedule:")
decoded_schedule, final_makespan = decode_chromosome(best_chrom, orders)
for op in decoded_schedule:
    print(f"Order {op[0]} Operation {op[1]} on Machine {op[2]}: start at {op[3]}, finish at {op[4]}")
print("Final Makespan:", final_makespan)