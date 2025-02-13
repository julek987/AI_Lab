import pandas as pd
import random

# Read the Excel file.
df = pd.read_excel('GA_task.xlsx', header=None)
orders = []

# Process columns in pairs (each order is represented by 2 columns).
for col in range(0, df.shape[1], 2):
    order = []
    # Start from row 2 (skip header rows) to read operations.
    for i in range(2, df.shape[0]):
        # If both machine and duration values exist, add them as a tuple.
        if pd.notna(df.iloc[i, col]) and pd.notna(df.iloc[i, col + 1]):
            order.append((df.iloc[i, col], df.iloc[i, col + 1]))
    orders.append(order)


def create_chromosome(orders):
    # Create a chromosome that encodes the sequence of operations.
    chromosome = []
    # For each order, add its ID as many times as there are operations.
    for order_id, order in enumerate(orders):
        chromosome += [order_id] * len(order)
    # Randomize the sequence.
    random.shuffle(chromosome)
    return chromosome

chromosome = create_chromosome(orders)
print("Chromosome:", chromosome)


def decode_chromosome(chromosome, orders):
    # Track machine availability times.
    machine_available = {}
    # Track the time when each order is ready (after its last scheduled operation).
    order_ready_time = [0] * len(orders)
    # Track the next operation index for each order.
    next_op_index = [0] * len(orders)

    schedule = []

    # Go through the chromosome (sequence of order IDs).
    for order_id in chromosome:
        op_index = next_op_index[order_id]
        # Skip if all operations for this order are already scheduled.
        if op_index >= len(orders[order_id]):
            continue  # Extra gene; may occur if the chromosome isn't repaired.

        # Get the machine and duration for the next operation of the order.
        machine, duration = orders[order_id][op_index]
        # Operation starts when both the order and machine are ready.
        start_time = max(order_ready_time[order_id], machine_available.get(machine, 0))
        finish_time = start_time + duration

        # Record the scheduled operation.
        schedule.append((order_id, op_index, machine, start_time, finish_time))

        # Update the ready times for the order and machine.
        order_ready_time[order_id] = finish_time
        machine_available[machine] = finish_time

        # Move to the next operation in the order.
        next_op_index[order_id] += 1

    # The makespan is the maximum completion time among all orders.
    makespan = max(order_ready_time)
    return schedule, makespan


# Decode the chromosome to get the schedule and makespan.
schedule, makespan = decode_chromosome(chromosome, orders)
print("Schedule:")
for op in schedule:
    print(f"Order {op[0]} Operation {op[1]} on Machine {op[2]}: start at {op[3]}, finish at {op[4]}")
print("Makespan:", makespan)


def create_population(orders, pop_size):
    # Create a list of chromosomes to form the initial population.
    return [create_chromosome(orders) for _ in range(pop_size)]


def evaluate_population(population, orders):
    # Evaluate each chromosome by decoding it and returning its makespan.
    fitnesses = []
    for chromosome in population:
        _, makespan = decode_chromosome(chromosome, orders)
        fitnesses.append(makespan)
    return fitnesses


def tournament_selection(population, fitnesses, tournament_size):
    # Randomly pick a group (tournament) and return the best chromosome.
    participants = random.sample(list(zip(population, fitnesses)), tournament_size)
    best = min(participants, key=lambda x: x[1])
    return best[0]


def crossover(parent1, parent2):
    length = len(parent1)
    jobs = set(parent1)
    # Build dictionaries mapping each order (job) to a list of its positions in the parent chromosomes.
    positions1 = {job: [] for job in jobs}
    positions2 = {job: [] for job in jobs}

    for i, gene in enumerate(parent1):
        positions1[gene].append(i)
    for i, gene in enumerate(parent2):
        positions2[gene].append(i)

    candidate_positions = []
    # For each order, compute the average position from both parents.
    for job in jobs:
        occ1 = positions1[job]
        occ2 = positions2[job]
        occ1.sort()
        occ2.sort()
        for i in range(len(occ1)):
            avg = (occ1[i] + occ2[i]) / 2.0
            candidate_positions.append((avg, job))

    # Sort orders by their averaged positions to form the child chromosome.
    candidate_positions.sort(key=lambda x: x[0])
    child = [job for _, job in candidate_positions]
    return child


def mutate(chromosome, mutation_rate):
    # Create a copy of the chromosome for mutation.
    mutated = chromosome.copy()
    # For each gene, swap it with another randomly with a chance of mutation_rate.
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def genetic_algorithm(orders,
                      pop_size,
                      generations,
                      crossover_rate,
                      mutation_rate,
                      tournament_size):

    # Initialize the population.
    population = create_population(orders, pop_size)
    best_chromosome = None
    best_fitness = float('inf')

    for gen in range(generations):
        # Evaluate the population's fitness (makespan).
        fitnesses = evaluate_population(population, orders)

        # Track the best chromosome so far.
        for i, fit in enumerate(fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_chromosome = population[i]

        new_population = []
        # Create a new population using selection, crossover, and mutation.
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            # Apply crossover with a given probability.
            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutate the child.
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        print(f"Generation {gen + 1}: Best Makespan = {best_fitness}")

    return best_chromosome, best_fitness


# Run the genetic algorithm with chosen parameters.
best_chrom, best_fit = genetic_algorithm(orders,
                                         pop_size=150,
                                         generations=50,
                                         crossover_rate=0.6,
                                         mutation_rate=0.01,
                                         tournament_size=30)

print("\nBest Found Schedule:")
# Decode the best chromosome to get the final schedule.
decoded_schedule, final_makespan = decode_chromosome(best_chrom, orders)
for op in decoded_schedule:
    print(f"Order {op[0]} Operation {op[1]} on Machine {op[2]}: start at {op[3]}, finish at {op[4]}")
print("Final Makespan:", final_makespan)
