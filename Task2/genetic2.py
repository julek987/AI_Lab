import pandas as pd

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
