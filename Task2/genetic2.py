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


