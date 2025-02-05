import pandas as pd
import random

df = pd.read_excel('GA_task.xlsx', header=None)
orders = []

for col in range(0, df.shape[1], 2):
    order = []

    for i in range(2, df.shape[0]):
        if pd.notna(df.iloc[i, col]) and pd.notna(df.iloc[i, col + 1]):
            order.append((df.iloc[i, col], df.iloc[i, col + 1]))

    orders.append(order)



def create_chromosome(orders):
    chromosome = []
    for order_id, order in enumerate(orders):
        chromosome += [order_id] * len(order)
    random.shuffle(chromosome)
    return chromosome

chromosome = create_chromosome(orders)
print("Chromosome:", chromosome)


def decode_chromosome(chromosome, orders):

    # For each machine, track when it becomes available.
    machine_available = {}
    # For each order, track when the previous operation finished.
    order_ready_time = [0] * len(orders)
    # For each order, track which operation is next (initially 0 for all orders).
    next_op_index = [0] * len(orders)

    schedule = []

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


def create_population(orders, pop_size):
    return [create_chromosome(orders) for _ in range(pop_size)]


def evaluate_population(population, orders):
    fitnesses = []
    for chromosome in population:
        _, makespan = decode_chromosome(chromosome, orders)
        fitnesses.append(makespan)
    return fitnesses


def tournament_selection(population, fitnesses, tournament_size=3):
    participants = random.sample(list(zip(population, fitnesses)), tournament_size)
    best = min(participants, key=lambda x: x[1])
    return best[0]


def crossover(parent1, parent2):
    length = len(parent1)
    jobs = set(parent1)
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


def mutate(chromosome, mutation_rate=0.1):
    mutated = chromosome.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def genetic_algorithm(orders,
                      pop_size=50,
                      generations=100,
                      crossover_rate=0.8,
                      mutation_rate=0.1,
                      tournament_size=3):

    population = create_population(orders, pop_size)
    best_chromosome = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitnesses = evaluate_population(population, orders)

        for i, fit in enumerate(fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_chromosome = population[i]

        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            if random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutate(child, mutation_rate)

            new_population.append(child)

        population = new_population
        print(f"Generation {gen + 1}: Best Makespan = {best_fitness}")

    return best_chromosome, best_fitness


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