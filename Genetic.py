import numpy as np

# Step 1: Fitness Function
def fitness_function(x):
    return x ** 2  # Maximizing f(x) = x²

# Step 2: Initialize Population
def initialize_population(pop_size, bit_length):
    population = np.random.randint(0, 2, (pop_size, bit_length))  # Random binary population
    return population

# Step 3: Selection (Using Roulette Wheel Selection)
def selection(population, fitness):
    total_fitness = np.sum(fitness)
    avg_fitness = np.mean(fitness)
    
    # Calculate expected output as f(x) / avg(f(x)) and round it
    expected_output = np.round(fitness / avg_fitness)
    
    # Selecting individuals based on fitness (Roulette wheel selection)
    probabilities = fitness / total_fitness
    selected_population = population[np.random.choice(population.shape[0], size=population.shape[0], p=probabilities)]
    
    return selected_population, expected_output

# Step 4: Crossover (Single Point Crossover)
def crossover(population, crossover_point):
    offspring = []
    for i in range(0, len(population), 2):  # Pairing parents
        parent1 = population[i]
        parent2 = population[i + 1]
        
        # Perform single point crossover at crossover_point
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        offspring.extend([child1, child2])
    return np.array(offspring)

# Step 5: Mutation (Bit Flip Mutation)
def mutation(population, mutation_rate):
    mutated_population = population.copy()
    for i in range(len(population)):
        if np.random.rand() < mutation_rate:  # Mutation chance
            mutation_point = np.random.randint(0, len(population[i]))
            mutated_population[i][mutation_point] = 1 - mutated_population[i][mutation_point]
    return mutated_population

# Convert binary chromosome to decimal (x)
def binary_to_decimal(binary):
    return int(''.join(map(str, binary)), 2)

# Main Genetic Algorithm
def genetic_algorithm(pop_size=4, bit_length=5, generations=5, mutation_rate=0.05, crossover_point=4):
    population = initialize_population(pop_size, bit_length)
    
    for generation in range(generations):
        print(f"\nGeneration {generation + 1}:")
        
        # Calculate Fitness for the population
        decimal_values = np.array([binary_to_decimal(ind) for ind in population])
        fitness_values = fitness_function(decimal_values)
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        
        print(f"Population:\n{population}")
        print(f"Fitness Values: {fitness_values}")
        print(f"Avg Fitness: {avg_fitness}, Max Fitness: {max_fitness}")
        
        # Selection
        selected_population, expected_output = selection(population, fitness_values)
        
        # Crossover
        offspring = crossover(selected_population, crossover_point)
        
        # Mutation
        mutated_population = mutation(offspring, mutation_rate)
        
        # Recalculate fitness for new generation
        new_decimal_values = np.array([binary_to_decimal(ind) for ind in mutated_population])
        new_fitness_values = fitness_function(new_decimal_values)
        
        print(f"Offspring after Crossover:\n{offspring}")
        print(f"Mutated Population:\n{mutated_population}")
        print(f"New Fitness Values: {new_fitness_values}")
        
        # Expected Output Calculation (f(x) / Avg(f(x))) and round it
        expected_output_percentage = new_fitness_values / avg_fitness
        print(f"Expected Output: {expected_output_percentage}")
        
        # Actual Selection based on fitness
        actual_selection = np.random.choice(new_decimal_values, size=pop_size, p=new_fitness_values/np.sum(new_fitness_values))
        print(f"Actual Selection (from new population): {actual_selection}")
        
        # Update population for the next generation
        population = mutated_population
    
    return population

# Run the Genetic Algorithm
final_population = genetic_algorithm()

print(f"\nFinal Population:\n{final_population}")
