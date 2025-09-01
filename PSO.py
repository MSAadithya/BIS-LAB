import numpy as np

# Objective function
def f(x, y):
    return x**2 + y**2

# PSO parameters
num_particles = 20
num_iterations = 3

w = 0.7        # inertia
c1 = 1.5       # cognitive (self)
c2 = 1.5       # social (swarm)

# Initialize particles randomly in [-10,10]
positions = np.random.uniform(-10, 10, (num_particles, 2))
velocities = np.random.uniform(-1, 1, (num_particles, 2))

# Initialize personal bests
pbest_positions = positions.copy()
pbest_values = np.array([f(x, y) for x, y in positions])

# Initialize global best
gbest_index = np.argmin(pbest_values)
gbest_position = pbest_positions[gbest_index].copy()
gbest_value = pbest_values[gbest_index]

# Main PSO loop
for iteration in range(num_iterations):
    for i in range(num_particles):
        # Evaluate fitness
        fitness = f(positions[i][0], positions[i][1])

        # Update personal best
        if fitness < pbest_values[i]:
            pbest_values[i] = fitness
            pbest_positions[i] = positions[i].copy()

        # Update global best
        if fitness < gbest_value:
            gbest_value = fitness
            gbest_position = positions[i].copy()

    # Update velocity and position
    r1, r2 = np.random.rand(), np.random.rand()
    velocities = (w * velocities +
                  c1 * r1 * (pbest_positions - positions) +
                  c2 * r2 * (gbest_position - positions))
    positions += velocities

    print(f"Iteration {iteration+1} -> Best Value: {gbest_value:.6f}, Best Position: {gbest_position}")

print("\nFinal Solution:")
print("Best Position:", gbest_position)
print("Best Value:", gbest_value)
