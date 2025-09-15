import numpy as np
import random

# Knapsack Problem setup
values = [24,13,23,15,16]  # profit of items
weights = [12,7,11,8,9]   # weight of items
capacity = 26
n_items = len(values)

# Cuckoo Search Parameters
n = 10        # number of nests
pa = 0.25     # probability of discovery
Maxt = 100    # max iterations

# Fitness function
def fitness(solution):
    total_value = sum(v * s for v, s in zip(values, solution))
    total_weight = sum(w * s for w, s in zip(weights, solution))
    if total_weight > capacity:
        return 0  # penalize infeasible solutions
    return total_value

# Generate random binary solution
def random_solution():
    return [random.randint(0, 1) for _ in range(n_items)]

# Levy flight step
def levy_flight(Lambda):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
            (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, size=n_items)
    v = np.random.normal(0, 1, size=n_items)
    step = u / abs(v) ** (1 / Lambda)
    return step

# Cuckoo Search Algorithm
def cuckoo_search():
    nests = [random_solution() for _ in range(n)]
    fitness_values = [fitness(sol) for sol in nests]
    
    best_index = np.argmax(fitness_values)
    best_solution = nests[best_index]
    best_fitness = fitness_values[best_index]
    
    for t in range(Maxt):
        # Generate new solution by Levy flight
        new_solution = nests[random.randint(0, n-1)].copy()
        step = levy_flight(1.5)  # Levy distribution parameter
        for i in range(n_items):
            if random.random() < abs(step[i]) % 1:  # random perturbation
                new_solution[i] = 1 - new_solution[i]  # flip bit
        
        # Evaluate new solution
        f_new = fitness(new_solution)
        
        # Choose a random nest to compare
        j = random.randint(0, n-1)
        if f_new > fitness_values[j]:
            nests[j] = new_solution
            fitness_values[j] = f_new
        
        # Abandon some nests (Discovery by host birds)
        for i in range(n):
            if random.random() < pa:
                nests[i] = random_solution()
                fitness_values[i] = fitness(nests[i])
        
        # Update best solution
        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > best_fitness:
            best_solution = nests[best_index]
            best_fitness = fitness_values[best_index]
    
    return best_solution, best_fitness

# Run algorithm
best_sol, best_fit = cuckoo_search()
print("Best Solution:", best_sol)
print("Total Value:", best_fit)
print("Total Weight:", sum(w*s for w,s in zip(weights, best_sol)))
