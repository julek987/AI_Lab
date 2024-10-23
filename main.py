from City import City, create_random_city
from Graph import Graph
from State import State

from collections import deque

def bfs(graph, start_city):
    # Queue for BFS, containing the initial state (starting from start_city)
    initial_state = State(start_city, {start_city}, 0)
    queue = deque([initial_state])

    n = len(graph.graph)  # Total number of cities
    best_cost = float('inf')
    best_path = None

    while queue:
        current_state = queue.popleft()

        # If all cities have been visited, complete the cycle by returning to the start city
        if len(current_state.visited) == n:
            final_cost = current_state.total_cost + current_state.current_city.distance_to(start_city)
            if final_cost < best_cost:
                best_cost = final_cost
                best_path = current_state.visited
            continue

        # Expand the current node by visiting each unvisited city
        for neighbor, travel_cost in graph.graph[current_state.current_city]:
            if neighbor not in current_state.visited:
                new_visited = current_state.visited | {neighbor}
                new_total_cost = current_state.total_cost + travel_cost
                new_state = State(neighbor, new_visited, new_total_cost)
                queue.append(new_state)

    return best_path, best_cost

def dfs(graph, start_city):
    # Stack for DFS, containing the initial state (starting from start_city)
    initial_state = State(start_city, {start_city}, 0)
    stack = [initial_state]

    n = len(graph.graph)  # Total number of cities
    best_cost = float('inf')
    best_path = None

    while stack:
        current_state = stack.pop()

        # If all cities have been visited, complete the cycle by returning to the start city
        if len(current_state.visited) == n:
            final_cost = current_state.total_cost + current_state.current_city.distance_to(start_city)
            if final_cost < best_cost:
                best_cost = final_cost
                best_path = current_state.visited
            continue

        # Expand the current node by visiting each unvisited city
        for neighbor, travel_cost in graph.graph[current_state.current_city]:
            if neighbor not in current_state.visited:
                new_visited = current_state.visited | {neighbor}
                new_total_cost = current_state.total_cost + travel_cost
                new_state = State(neighbor, new_visited, new_total_cost)
                stack.append(new_state)

    return best_path, best_cost

def nearest_neighbour(graph, start_city):
    # Start from the initial city
    current_city = start_city
    visited = {current_city}  # Keep track of visited cities
    total_cost = 0
    path = [current_city]

    n = len(graph.graph)  # Total number of cities

    while len(visited) < n:
        # Find the nearest unvisited city
        nearest_city = None
        min_cost = float('inf')

        for neighbor, travel_cost in graph.graph[current_city]:
            if neighbor not in visited and travel_cost < min_cost:
                nearest_city = neighbor
                min_cost = travel_cost

        # Move to the nearest city
        if nearest_city is not None:
            path.append(nearest_city)
            visited.add(nearest_city)
            total_cost += min_cost
            current_city = nearest_city

    # Return to the starting city to complete the cycle
    total_cost += current_city.distance_to(start_city)
    path.append(start_city)

    return path, total_cost


city1 = create_random_city("CityA")
city2 = create_random_city("CityB")
city3 = create_random_city("CityC")
city4 = create_random_city("CityD")
city5 = create_random_city("CityE")

# Symmetrical scenario (all direct connections and symmetrical cost)
city_graph_symmetrical = Graph()
city_graph_symmetrical.add_city(city1)
city_graph_symmetrical.add_city(city2)
city_graph_symmetrical.add_city(city3)
city_graph_symmetrical.add_city(city4)
city_graph_symmetrical.add_city(city5)

city_graph_symmetrical.add_road(city1, city2, asymmetrical=False)
city_graph_symmetrical.add_road(city1, city3, asymmetrical=False)
city_graph_symmetrical.add_road(city1, city4, asymmetrical=False)
city_graph_symmetrical.add_road(city1, city5, asymmetrical=False)

city_graph_symmetrical.add_road(city2, city3, asymmetrical=False)
city_graph_symmetrical.add_road(city2, city4, asymmetrical=False)
city_graph_symmetrical.add_road(city2, city5, asymmetrical=False)

city_graph_symmetrical.add_road(city3, city4, asymmetrical=False)
city_graph_symmetrical.add_road(city3, city5, asymmetrical=False)

city_graph_symmetrical.add_road(city4, city5, asymmetrical=False)

# Display the graph for the symmetrical scenario
print("Symmetrical graph:")
city_graph_symmetrical.display_graph()

# Run BFS and DFS for symmetrical scenario
best_path_sym, best_cost_sym = bfs(city_graph_symmetrical, city1)
print(f"BFS Best Path (Symmetrical): {best_path_sym}, Cost: {best_cost_sym}")

best_path_sym, best_cost_sym = dfs(city_graph_symmetrical, city1)
print(f"DFS Best Path (Symmetrical): {best_path_sym}, Cost: {best_cost_sym}")

# Run Nearest Neighbour for symmetrical scenario
best_path_nn_sym, best_cost_nn_sym = nearest_neighbour(city_graph_symmetrical, city1)
print(f"Nearest Neighbour Best Path (Symmetrical): {best_path_nn_sym}, Cost: {best_cost_nn_sym}")

# =======================================================
# Asymmetrical scenario (all direct connections but asymmetrical cost)
city_graph_asymmetrical = Graph()
city_graph_asymmetrical.add_city(city1)
city_graph_asymmetrical.add_city(city2)
city_graph_asymmetrical.add_city(city3)
city_graph_asymmetrical.add_city(city4)
city_graph_asymmetrical.add_city(city5)

city_graph_asymmetrical.add_road(city1, city2, asymmetrical=True)
city_graph_asymmetrical.add_road(city1, city3, asymmetrical=True)
city_graph_asymmetrical.add_road(city1, city4, asymmetrical=True)
city_graph_asymmetrical.add_road(city1, city5, asymmetrical=True)

city_graph_asymmetrical.add_road(city2, city1, asymmetrical=True)
city_graph_asymmetrical.add_road(city2, city3, asymmetrical=True)
city_graph_asymmetrical.add_road(city2, city4, asymmetrical=True)
city_graph_asymmetrical.add_road(city2, city5, asymmetrical=True)

city_graph_asymmetrical.add_road(city3, city1, asymmetrical=True)
city_graph_asymmetrical.add_road(city3, city2, asymmetrical=True)
city_graph_asymmetrical.add_road(city3, city4, asymmetrical=True)
city_graph_asymmetrical.add_road(city3, city5, asymmetrical=True)

city_graph_asymmetrical.add_road(city4, city1, asymmetrical=True)
city_graph_asymmetrical.add_road(city4, city2, asymmetrical=True)
city_graph_asymmetrical.add_road(city4, city3, asymmetrical=True)
city_graph_asymmetrical.add_road(city4, city5, asymmetrical=True)

city_graph_asymmetrical.add_road(city5, city1, asymmetrical=True)
city_graph_asymmetrical.add_road(city5, city2, asymmetrical=True)
city_graph_asymmetrical.add_road(city5, city3, asymmetrical=True)
city_graph_asymmetrical.add_road(city5, city4, asymmetrical=True)

# Display the graph for the asymmetrical scenario
print("\nAsymmetrical graph:")
city_graph_asymmetrical.display_graph()

# Run BFS and DFS for asymmetrical scenario
best_path_asym, best_cost_asym = bfs(city_graph_asymmetrical, city1)
print(f"BFS Best Path (Asymmetrical): {best_path_asym}, Cost: {best_cost_asym}")

best_path_asym, best_cost_asym = dfs(city_graph_asymmetrical, city1)
print(f"DFS Best Path (Asymmetrical): {best_path_asym}, Cost: {best_cost_asym}")

# Run Nearest Neighbour for asymmetrical scenario
best_path_nn_asym, best_cost_nn_asym = nearest_neighbour(city_graph_asymmetrical, city1)
print(f"Nearest Neighbour Best Path (Asymmetrical): {best_path_nn_asym}, Cost: {best_cost_nn_asym}")

# =======================================================
# Symmetrical scenario (80% connections and symmetrical cost)
city_graph_symmetrical_80 = Graph()
city_graph_symmetrical_80.add_city(city1)
city_graph_symmetrical_80.add_city(city2)
city_graph_symmetrical_80.add_city(city3)
city_graph_symmetrical_80.add_city(city4)
city_graph_symmetrical_80.add_city(city5)

city_graph_symmetrical_80.add_road(city1, city2, asymmetrical=False)
city_graph_symmetrical_80.add_road(city1, city3, asymmetrical=False)

city_graph_symmetrical_80.add_road(city2, city4, asymmetrical=False)
city_graph_symmetrical_80.add_road(city2, city5, asymmetrical=False)

city_graph_symmetrical_80.add_road(city3, city2, asymmetrical=False)
city_graph_symmetrical_80.add_road(city3, city5, asymmetrical=False)

city_graph_symmetrical_80.add_road(city4, city1, asymmetrical=False)
city_graph_symmetrical_80.add_road(city4, city5, asymmetrical=False)

# Display the graph for the symmetrical scenario with 80% connections
print("\nSymmetrical graph (80%):")
city_graph_symmetrical_80.display_graph()

# Run BFS and DFS for symmetrical scenario with 80% scenarios
best_path_asym, best_cost_asym = bfs(city_graph_symmetrical_80, city1)
print(f"BFS Best Path (Symmetrical): {best_path_asym}, Cost: {best_cost_asym}")

best_path_asym, best_cost_asym = dfs(city_graph_symmetrical_80, city1)
print(f"DFS Best Path (Symmetrical): {best_path_asym}, Cost: {best_cost_asym}")

# Run Nearest Neighbour for symmetrical scenario with 80% connections
best_path_nn_sym, best_cost_nn_sym = nearest_neighbour(city_graph_symmetrical_80, city1)
print(f"Nearest Neighbour Best Path (Symmetrical): {best_path_nn_sym}, Cost: {best_cost_nn_sym}")


# =======================================================
# Asymmetrical scenario (80% connections and asymmetrical cost)
city_graph_asymmetrical_80 = Graph()
city_graph_asymmetrical_80.add_city(city1)
city_graph_asymmetrical_80.add_city(city2)
city_graph_asymmetrical_80.add_city(city3)
city_graph_asymmetrical_80.add_city(city4)
city_graph_asymmetrical_80.add_city(city5)

city_graph_asymmetrical_80.add_road(city1, city2, asymmetrical=True)
city_graph_asymmetrical_80.add_road(city1, city3, asymmetrical=True)

city_graph_asymmetrical_80.add_road(city2, city4, asymmetrical=True)
city_graph_asymmetrical_80.add_road(city2, city5, asymmetrical=True)

city_graph_asymmetrical_80.add_road(city3, city2, asymmetrical=True)
city_graph_asymmetrical_80.add_road(city3, city5, asymmetrical=True)

city_graph_asymmetrical_80.add_road(city4, city1, asymmetrical=True)
city_graph_asymmetrical_80.add_road(city4, city5, asymmetrical=True)

city_graph_asymmetrical_80.add_road(city5, city2, asymmetrical=True)

# Display the graph for the symmetrical scenario with 80% connections
print("\nAsymmetrical graph (80%):")
city_graph_asymmetrical_80.display_graph()

# Run BFS and DFS for symmetrical scenario with 80% connections
best_path_asym, best_cost_asym = bfs(city_graph_asymmetrical_80, city1)
print(f"BFS Best Path (Asymmetrical): {best_path_asym}, Cost: {best_cost_asym}")

best_path_asym, best_cost_asym = dfs(city_graph_asymmetrical_80, city1)
print(f"DFS Best Path (Asymmetrical): {best_path_asym}, Cost: {best_cost_asym}")

# Run Nearest Neighbour for asymmetrical scenario with 80% connections
best_path_nn_asym, best_cost_nn_asym = nearest_neighbour(city_graph_asymmetrical_80, city1)
print(f"Nearest Neighbour Best Path (Asymmetrical): {best_path_nn_asym}, Cost: {best_cost_nn_asym}")

print("test")

#TODO
#In DFS and BFS - full path not showing
#Vulnerable to last visited city not being connected to starting city

