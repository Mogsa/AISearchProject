import os
import sys
import time
import random
from datetime import datetime
import math
import itertools
import csv

# Function to run a single ACO instance with the given alpha and beta values
def run_aco_with_params(alpha, beta, city_file, num_trials=3, max_it=100, num_ants=20):
    """
    Run ACO algorithm with specific alpha and beta parameters.
    Returns the best tour length found.
    """
    print(f"Testing alpha={alpha:.2f}, beta={beta:.2f} on {city_file}...")
    
    # Read city file and build distance matrix
    # This is simplified - you can use your existing file loading code
    path_to_city_file = os.path.join("..", "city-files", city_file)
    
    # Import your existing city file reading and distance matrix creation code here
    # For brevity, we're assuming the matrix is already created
    
    # We'll collect multiple runs to account for randomness
    results = []
    
    for trial in range(num_trials):
        # ACO Core Algorithm (simplified version)
        
        # Initialize parameters
        rho = 0.15
        q0 = 0.9
        
        # Initial tour using NN heuristic for tau bounds
        seed_tour = nearest_neighbour(dist_matrix)
        seed_length = calculate_tour_length(seed_tour, dist_matrix)
        
        # Setup pheromone matrix with appropriate bounds
        tau_max = 1.0 / (rho * seed_length)
        pheromone = [[tau_max for _ in range(num_cities)] for _ in range(num_cities)]
        
        # Run ACO algorithm with the given alpha and beta
        best_length = run_single_aco_trial(alpha, beta, pheromone, dist_matrix, 
                                           num_cities, max_it, num_ants, rho, q0)
        
        results.append(best_length)
    
    # Return average tour length across trials
    avg_length = sum(results) / len(results)
    best_length = min(results)
    
    print(f"Alpha={alpha:.2f}, Beta={beta:.2f}: Avg={avg_length:.2f}, Best={best_length}")
    return avg_length, best_length

def run_single_aco_trial(alpha, beta, pheromone, dist_matrix, num_cities, 
                         max_it, num_ants, rho, q0):
    """Core ACO algorithm implementation for a single trial"""
    
    # Create visibility matrix
    visibility = [[0 if i==j or dist_matrix[i][j] == 0 else 1.0/dist_matrix[i][j]
                  for j in range(num_cities)] for i in range(num_cities)]
    
    # Initialize best tour
    best_tour = None
    best_length = float('inf')
    
    # Main ACO loop
    for it in range(max_it):
        # Ant tours construction
        for ant in range(num_ants):
            tour = construct_tour(pheromone, visibility, alpha, beta, q0, dist_matrix, num_cities)
            tour_length = calculate_tour_length(tour, dist_matrix)
            
            # Update best tour
            if tour_length < best_length:
                best_length = tour_length
                best_tour = tour.copy()
        
        # Pheromone update
        # Evaporation
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1 - rho)
        
        # Deposit from best tour
        delta = 1.0 / best_length
        for i in range(num_cities - 1):
            a, b = best_tour[i], best_tour[i+1]
            pheromone[a][b] += delta
            pheromone[b][a] = pheromone[a][b]  # Symmetric
        
        # Close the loop
        a, b = best_tour[-1], best_tour[0]
        pheromone[a][b] += delta
        pheromone[b][a] = pheromone[a][b]  # Symmetric
    
    return best_length

def construct_tour(pheromone, visibility, alpha, beta, q0, dist_matrix, num_cities):
    """Construct a single ant tour using ACO rules"""
    start_city = random.randrange(num_cities)
    visited = [False] * num_cities
    visited[start_city] = True
    tour = [start_city]
    current = start_city
    
    while len(tour) < num_cities:
        if random.random() < q0:
            # Exploitation (choose best)
            next_city = -1
            max_val = -1
            
            for j in range(num_cities):
                if not visited[j]:
                    # Ensure safe division
                    vis = visibility[current][j] if visibility[current][j] > 0 else 1e-10
                    val = (pheromone[current][j]**alpha) * (vis**beta)
                    
                    if val > max_val:
                        max_val = val
                        next_city = j
        else:
            # Exploration (probabilistic)
            probs = []
            total = 0
            
            for j in range(num_cities):
                if not visited[j]:
                    # Ensure safe division
                    vis = visibility[current][j] if visibility[current][j] > 0 else 1e-10
                    val = (pheromone[current][j]**alpha) * (vis**beta)
                    probs.append((j, val))
                    total += val
            
            # Roulette wheel selection
            if total > 0:
                r = random.random() * total
                cum_prob = 0
                next_city = -1
                
                for j, val in probs:
                    cum_prob += val
                    if cum_prob >= r:
                        next_city = j
                        break
                
                # Failsafe
                if next_city == -1 and probs:
                    next_city = probs[-1][0]
            else:
                # If all probabilities are zero, choose randomly
                candidates = [j for j in range(num_cities) if not visited[j]]
                next_city = random.choice(candidates) if candidates else -1
        
        if next_city != -1:
            tour.append(next_city)
            visited[next_city] = True
            current = next_city
        else:
            # Failsafe: if somehow no city was chosen, pick randomly
            candidates = [j for j in range(num_cities) if not visited[j]]
            if candidates:
                next_city = random.choice(candidates)
                tour.append(next_city)
                visited[next_city] = True
                current = next_city
            else:
                break  # No more cities to visit
    
    return tour

def calculate_tour_length(tour, dist_matrix):
    """Calculate the total length of a tour"""
    length = 0
    num_cities = len(tour)
    
    for i in range(num_cities - 1):
        length += dist_matrix[tour[i]][tour[i+1]]
    
    # Add return to starting city
    length += dist_matrix[tour[-1]][tour[0]]
    
    return length

def nearest_neighbour(dist_matrix):
    """Generate a tour using nearest neighbor heuristic"""
    num_cities = len(dist_matrix)
    start_city = random.randrange(num_cities)
    
    tour = [start_city]
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)
    
    current_city = start_city
    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return tour

def grid_search(city_file, alphas, betas, trials_per_combo=3):
    """
    Perform a grid search to find the best alpha and beta parameters
    for the given city file.
    """
    results = []
    
    # Open a CSV file to log results
    with open(f"alpha_beta_search_{city_file.replace('.txt', '')}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Alpha", "Beta", "Avg Length", "Best Length"])
        
        # Try each alpha-beta combination
        for alpha, beta in itertools.product(alphas, betas):
            avg_length, best_length = run_aco_with_params(
                alpha, beta, city_file, trials_per_combo)
            
            results.append((alpha, beta, avg_length, best_length))
            
            # Log to CSV
            csvwriter.writerow([alpha, beta, avg_length, best_length])
            csvfile.flush()  # Ensure data is written immediately
    
    # Sort by average tour length
    results.sort(key=lambda x: x[2])
    
    print("\nTop 5 Alpha-Beta Combinations:")
    for i, (alpha, beta, avg_length, best_length) in enumerate(results[:5], 1):
        print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}: Avg={avg_length:.2f}, Best={best_length}")
    
    return results

def random_search(city_file, num_trials=50, alpha_range=(0.0, 5.0), beta_range=(0.0, 10.0)):
    """
    Perform a random search to find good alpha and beta parameters
    for the given city file.
    """
    results = []
    
    # Open a CSV file to log results
    with open(f"alpha_beta_random_search_{city_file.replace('.txt', '')}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Alpha", "Beta", "Avg Length", "Best Length"])
        
        # Generate and try random alpha-beta combinations
        for _ in range(num_trials):
            alpha = random.uniform(*alpha_range)
            beta = random.uniform(*beta_range)
            
            avg_length, best_length = run_aco_with_params(
                alpha, beta, city_file, trials_per_combo=2)
            
            results.append((alpha, beta, avg_length, best_length))
            
            # Log to CSV
            csvwriter.writerow([alpha, beta, avg_length, best_length])
            csvfile.flush()  # Ensure data is written immediately
    
    # Sort by average tour length
    results.sort(key=lambda x: x[2])
    
    print("\nTop 5 Alpha-Beta Combinations from Random Search:")
    for i, (alpha, beta, avg_length, best_length) in enumerate(results[:5], 1):
        print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}: Avg={avg_length:.2f}, Best={best_length}")
    
    return results

def sequential_tuning(city_file, initial_alpha=1.0, initial_beta=2.0, steps=5, trials_per_step=3):
    """
    Perform sequential parameter tuning, first finding good alpha with fixed beta,
    then finding good beta with the best alpha.
    """
    # First phase: Find good alpha with fixed beta
    print("Phase 1: Finding best alpha with fixed beta...")
    alpha_range = [max(0.1, initial_alpha - 1.5), initial_alpha - 0.5, 
                  initial_alpha, initial_alpha + 0.5, min(5.0, initial_alpha + 1.5)]
    
    alpha_results = []
    for alpha in alpha_range:
        avg_length, best_length = run_aco_with_params(
            alpha, initial_beta, city_file, trials_per_step)
        alpha_results.append((alpha, avg_length, best_length))
    
    # Find best alpha
    alpha_results.sort(key=lambda x: x[1])  # Sort by average length
    best_alpha = alpha_results[0][0]
    
    print(f"\nBest alpha found: {best_alpha:.2f}")
    
    # Second phase: Find good beta with fixed best alpha
    print("\nPhase 2: Finding best beta with best alpha...")
    beta_range = [max(0.1, initial_beta - 1.5), initial_beta - 0.5, 
                initial_beta, initial_beta + 0.5, min(10.0, initial_beta + 1.5)]
    
    beta_results = []
    for beta in beta_range:
        avg_length, best_length = run_aco_with_params(
            best_alpha, beta, city_file, trials_per_step)
        beta_results.append((beta, avg_length, best_length))
    
    # Find best beta
    beta_results.sort(key=lambda x: x[1])  # Sort by average length
    best_beta = beta_results[0][0]
    
    print(f"\nBest beta found: {best_beta:.2f}")
    
    # Optionally, do a fine-grained search around the best values
    print("\nFine-tuning around best values...")
    fine_alpha_range = [max(0.1, best_alpha - 0.2), best_alpha - 0.1, 
                        best_alpha, best_alpha + 0.1, min(5.0, best_alpha + 0.2)]
    fine_beta_range = [max(0.1, best_beta - 0.4), best_beta - 0.2, 
                      best_beta, best_beta + 0.2, min(10.0, best_beta + 0.4)]
    
    fine_results = []
    for alpha, beta in itertools.product(fine_alpha_range, fine_beta_range):
        avg_length, best_length = run_aco_with_params(
            alpha, beta, city_file, trials_per_step)
        fine_results.append((alpha, beta, avg_length, best_length))
    
    # Sort and output final results
    fine_results.sort(key=lambda x: x[2])  # Sort by average length
    
    print("\nTop 5 Fine-Tuned Alpha-Beta Combinations:")
    for i, (alpha, beta, avg_length, best_length) in enumerate(fine_results[:5], 1):
        print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}: Avg={avg_length:.2f}, Best={best_length}")
    
    return fine_results[0][0], fine_results[0][1]  # Return best alpha and beta

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        city_file = sys.argv[1]
    else:
        city_file = "AISearchfile180.txt"  # Default city file
    
    print(f"Parameter search for ACO on city file: {city_file}")
    print("Choose search method:")
    print("1. Grid Search")
    print("2. Random Search")
    print("3. Sequential Tuning")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        print("Running Grid Search...")
        # Define grid search parameters
        alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        betas = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        grid_search(city_file, alphas, betas)
    
    elif choice == "2":
        print("Running Random Search...")
        random_search(city_file, num_trials=30)
    
    elif choice == "3":
        print("Running Sequential Tuning...")
        best_alpha, best_beta = sequential_tuning(city_file)
        print(f"\nFinal best parameters: Alpha={best_alpha:.2f}, Beta={best_beta:.2f}")
    
    else:
        print("Invalid choice. Exiting.")