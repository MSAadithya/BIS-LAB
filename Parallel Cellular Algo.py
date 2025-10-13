import numpy as np
import random

# Constants for simulation
NUM_CELLS = 10  # Number of cloud resource configurations (cells)
NUM_ITERATIONS = 100  # Number of iterations to improve the allocation
NUM_RESOURCES = 3  # CPU, Memory, and Storage
ALPHA = [0.5, 0.3, 0.2]  # Weight for each resource type: CPU, Memory, Storage
BETA = 0.1  # Weight for cost minimization
GAMMA = 0.1  # Weight for load imbalance minimization

# Initial resource allocation based on user's input
initial_state = np.array([
    [6, 4, 8],  # VM 1
    [7, 6, 6],  # VM 2
    [2, 3, 5],  # VM 3
    [5, 8, 3],  # VM 4
    [9, 5, 7],  # VM 5
    [3, 6, 5],  # VM 6
    [4, 7, 4],  # VM 7
    [8, 5, 6],  # VM 8
    [6, 7, 5],  # VM 9
    [7, 3, 9]   # VM 10
])

# Calculate efficiency for CPU, Memory, and Storage usage (simplified for demo)
def calculate_efficiency(resource_allocation):
    """
    Calculate resource efficiency (higher is better).
    A simple linear efficiency based on usage (for simplicity in this example).
    """
    cpu_efficiency = np.sum(resource_allocation[:, 0]) / (NUM_CELLS * 10)  # Max CPU = 10 per cell
    memory_efficiency = np.sum(resource_allocation[:, 1]) / (NUM_CELLS * 10)  # Max Memory = 10 per cell
    storage_efficiency = np.sum(resource_allocation[:, 2]) / (NUM_CELLS * 10)  # Max Storage = 10 per cell
    
    return cpu_efficiency, memory_efficiency, storage_efficiency

# Calculate the load imbalance (simple difference between max and min load)
def calculate_load_imbalance(resource_allocation):
    """
    Calculate load imbalance (the difference between the maximum and minimum resource usage).
    A high load imbalance indicates that resources are not distributed evenly.
    """
    total_load = np.sum(resource_allocation, axis=1)  # Sum across rows (VMs)
    max_load = np.max(total_load)  # Max load across all configurations
    min_load = np.min(total_load)  # Min load across all configurations
    return (max_load - min_load) / np.sum(total_load)  # Normalized imbalance

# Calculate the cost based on resource consumption (simplified)
def calculate_cost(resource_allocation):
    """
    Calculate the cost of the allocation configuration (e.g., energy, storage, etc.).
    Here we use a simple linear cost function for illustration.
    """
    cpu_cost = np.sum(resource_allocation[:, 0]) * 0.1  # CPU cost per unit resource
    memory_cost = np.sum(resource_allocation[:, 1]) * 0.2  # Memory cost per unit resource
    storage_cost = np.sum(resource_allocation[:, 2]) * 0.3  # Storage cost per unit resource
    
    return cpu_cost + memory_cost + storage_cost

# The objective function to optimize
def objective_function(resource_allocation):
    """
    Objective function to evaluate a given resource allocation configuration.
    The goal is to maximize efficiency while minimizing cost and load imbalance.
    """
    cpu_eff, memory_eff, storage_eff = calculate_efficiency(resource_allocation)
    cost = calculate_cost(resource_allocation)
    load_imbalance = calculate_load_imbalance(resource_allocation)
    
    # Objective is to maximize efficiency and minimize cost/load imbalance
    return ALPHA[0] * cpu_eff + ALPHA[1] * memory_eff + ALPHA[2] * storage_eff - BETA * cost - GAMMA * load_imbalance

# Update function: the core of the PCA algorithm
def update_state(resource_allocation, best_allocation, mutation_rate=0.1):
    """
    Update the resource allocation state based on the best neighbor's state and with a mutation.
    This update mimics the process of evolving the solution based on neighboring states.
    """
    new_allocation = resource_allocation.copy()
    
    # Select a random neighboring cell (simulate interaction)
    neighbor_idx = random.choice(range(NUM_CELLS))
    best_neighbor = best_allocation[neighbor_idx]
    
    # Update the current allocation to be closer to the best neighbor (evolutionary step)
    random_cell = random.randint(0, NUM_CELLS-1)
    new_allocation[random_cell] = best_neighbor + np.random.normal(0, mutation_rate, size=3)
    
    # Ensure resources are within a valid range (positive values and reasonable bounds)
    new_allocation = np.clip(new_allocation, 1, 10)
    
    return new_allocation

# PCA main loop to optimize resource allocation
def pca_optimization():
    resource_allocation = initial_state.copy()
    best_allocation = resource_allocation.copy()
    best_score = -np.inf
    
    for iteration in range(NUM_ITERATIONS):
        print(f"Iteration {iteration+1}/{NUM_ITERATIONS}")
        
        # Evaluate fitness for each cell (allocation configuration)
        scores = np.array([objective_function(state) for state in resource_allocation])
        
        # Track the best allocation configuration
        current_best_score = np.max(scores)
        current_best_allocation = resource_allocation[np.argmax(scores)]
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_allocation = current_best_allocation
        
        # Update the resource allocation using the best configuration found so far
        resource_allocation = np.array([update_state(resource_allocation, best_allocation) for _ in range(NUM_CELLS)])
    
    return best_allocation, best_score

# Run PCA to optimize cloud resource allocation
best_allocation, best_score = pca_optimization()

# Final output
print("\nBest Resource Allocation Configuration:")
print(best_allocation)
print("\nBest Score (Fitness):", best_score)

