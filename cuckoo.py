import numpy as np
import math

# -----------------------------
# Knapsack Problem definition
# -----------------------------
values = [60, 100, 120, 90, 30]   # item values
weights = [10, 20, 30, 25, 5]     # item weights
capacity = 50                     # knapsack capacity
n_items = len(values)

# -----------------------------
# Cuckoo Search Parameters
# -----------------------------
n_nests = 15        # population size
p_a = 0.25          # discovery probability (abandonment rate)
max_iter = 200      # maximum iterations


# -----------------------------
# Fitness Function
# -----------------------------
def fitness(solution):
    total_value = np.sum(np.array(values) * solution)
    total_weight = np.sum(np.array(weights) * solution)
    if total_weight <= capacity:
        return total_value
    else:
        return 0  # penalize infeasible solution


# -----------------------------
# Lévy Flight generator
# -----------------------------
def levy_flight(Lambda=1.5):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / Lambda)
    return step


# -----------------------------
# Generate New Solution via Lévy Flight
# -----------------------------
def get_cuckoo(solution):
    new_solution = solution.copy()
    step_size = int(abs(levy_flight()) * n_items) % n_items
    # flip random bits
    flip_indices = np.random.choice(range(n_items), size=max(1, step_size), replace=False)
    for idx in flip_indices:
        new_solution[idx] = 1 - new_solution[idx]
    return new_solution


# -----------------------------
# Initialize Population
# -----------------------------
nests = [np.random.randint(0, 2, n_items) for _ in range(n_nests)]
fitness_values = [fitness(sol) for sol in nests]

# -----------------------------
# Main Loop
# -----------------------------
best_solution = nests[np.argmax(fitness_values)]
best_fitness = max(fitness_values)

for _ in range(max_iter):
    # Generate new solutions (cuckoo eggs)
    for i in range(n_nests):
        new_sol = get_cuckoo(nests[i])
        f_new = fitness(new_sol)

        # Replace a random nest if better
        j = np.random.randint(n_nests)
        if f_new > fitness_values[j]:
            nests[j] = new_sol
            fitness_values[j] = f_new

    # Abandon a fraction of worst nests
    sorted_indices = np.argsort(fitness_values)
    n_abandon = int(p_a * n_nests)
    for k in sorted_indices[:n_abandon]:
        nests[k] = np.random.randint(0, 2, n_items)
        fitness_values[k] = fitness(nests[k])

    # Update best
    current_best_idx = np.argmax(fitness_values)
    if fitness_values[current_best_idx] > best_fitness:
        best_solution = nests[current_best_idx]
        best_fitness = fitness_values[current_best_idx]

# -----------------------------
# Output
# -----------------------------
print("Best solution (items picked):", best_solution)
print("Total value:", best_fitness)
print("Total weight:", np.sum(np.array(weights) * best_solution))
