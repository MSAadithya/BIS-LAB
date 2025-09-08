import random
import numpy as np
import matplotlib.pyplot as plt

# ==== Parameters ====
NUM_ANTS = 10
NUM_ITERATIONS = 100
ALPHA = 1         # Pheromone importance
BETA = 5          # Distance priority
EVAPORATION_RATE = 0.5
Q = 100           # Pheromone increase factor

# ==== TSP Data ====
# Example cities (coordinates)
cities = np.array([
    [0, 0],
    [1, 5],
    [5, 2],
    [6, 6],
    [8, 3],
    [7, 9],
    [2, 7],
    [3, 3]
])

num_cities = len(cities)

# ==== Distance Matrix ====
def compute_distance_matrix(cities):
    dist_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dist_matrix

distance_matrix = compute_distance_matrix(cities)

# ==== ACO ====
def initialize_pheromone_matrix(value=1.0):
    return np.ones((num_cities, num_cities)) * value

def probability(i, j, pheromone, distances, visited):
    if j in visited:
        return 0
    return (pheromone[i][j] ** ALPHA) * ((1.0 / distances[i][j]) ** BETA)

def choose_next_city(i, pheromone, distances, visited):
    probs = [probability(i, j, pheromone, distances, visited) for j in range(num_cities)]
    total = sum(probs)
    if total == 0:
        return random.choice([j for j in range(num_cities) if j not in visited])
    probs = [p / total for p in probs]
    return np.random.choice(range(num_cities), p=probs)

def ant_colony_optimization():
    pheromone = initialize_pheromone_matrix()
    best_path = None
    best_length = float('inf')

    for iteration in range(NUM_ITERATIONS):
        all_paths = []
        all_lengths = []

        for ant in range(NUM_ANTS):
            path = []
            visited = set()

            start = random.randint(0, num_cities - 1)
            path.append(start)
            visited.add(start)

            for _ in range(num_cities - 1):
                current = path[-1]
                next_city = choose_next_city(current, pheromone, distance_matrix, visited)
                path.append(next_city)
                visited.add(next_city)

            path.append(path[0])  # Return to start
            length = sum(distance_matrix[path[i]][path[i + 1]] for i in range(num_cities))
            all_paths.append(path)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_path = path

        # Evaporate pheromone
        pheromone *= (1 - EVAPORATION_RATE)

        # Deposit pheromone
        for path, length in zip(all_paths, all_lengths):
            for i in range(num_cities):
                from_city = path[i]
                to_city = path[i + 1]
                pheromone[from_city][to_city] += Q / length
                pheromone[to_city][from_city] += Q / length  # since it's undirected

        print(f"Iteration {iteration + 1}: Best length = {best_length:.2f}")

    return best_path, best_length

# ==== Run the Algorithm ====
best_path, best_length = ant_colony_optimization()
print("Best path found:", best_path)
print("Shortest distance:", best_length)

# ==== Plotting ====
def plot_path(cities, path):
    plt.figure(figsize=(8, 6))
    x = [cities[i][0] for i in path]
    y = [cities[i][1] for i in path]
    plt.plot(x, y, marker='o', linestyle='-')
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 0.1, yi + 0.1, f"City {i}", fontsize=9)
    plt.title("Best TSP Route Found by ACO")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

plot_path(cities, best_path)
