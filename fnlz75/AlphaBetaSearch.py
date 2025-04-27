import os
import sys
import time
import random
from datetime import datetime
import math
import itertools
import csv

# Function to run a single ACO instance with the given parameters
def run_aco_with_params(alpha, beta, city_file, rho=0.15, lambda_vis=0.5, 
                        candidate_list_size=20, num_trials=3, max_it=800, num_ants=25):
    """
    Run ACO algorithm with specific parameters.
    Returns the best tour length found and average performance.
    """
    print(f"Testing alpha={alpha:.2f}, beta={beta:.2f}, rho={rho:.2f}, lambda_vis={lambda_vis:.2f}, CL={candidate_list_size} on {city_file}...")
    
    # Read city file and build distance matrix using the standard code from AlgAenhanced
    path_to_city_file = os.path.join("..", "city-files", city_file)
    
    # Read city file (simplified - actual implementation should use the standard code)
    if not os.path.exists(path_to_city_file):
        print(f"Error: City file {path_to_city_file} not found.")
        return float('inf'), float('inf')
    
    # Track execution time
    start_time = time.time()
    
    # Parse file and build distance matrix (placeholder - actual implementation needed)
    # This would be replaced with the proper reading code as in AlgAenhanced.py
    ord_range = [[32, 126]]
    try:
        # This is simplified - the actual code would use the functions from AlgAenhanced
        file_string = read_file_into_string(path_to_city_file, ord_range)
        file_string = remove_all_spaces(file_string)
        
        location = file_string.find("SIZE=")
        if location == -1:
            print(f"Error: Invalid format in {city_file}")
            return float('inf'), float('inf')
            
        comma = file_string.find(",", location)
        if comma == -1:
            print(f"Error: Invalid format in {city_file}")
            return float('inf'), float('inf')
            
        num_cities_as_string = file_string[location + 5:comma]
        num_cities = integerize(num_cities_as_string)
        
        comma = comma + 1
        stripped_file_string = file_string[comma:]
        distances = convert_to_list_of_int(stripped_file_string)
        
        counted_distances = len(distances)
        if counted_distances == num_cities * num_cities:
            city_format = "full"
        elif counted_distances == (num_cities * (num_cities + 1))/2:
            city_format = "upper_tri"
        elif counted_distances == (num_cities * (num_cities - 1))/2:
            city_format = "strict_upper_tri"
        else:
            print(f"Error: Invalid format in {city_file}")
            return float('inf'), float('inf')
        
        dist_matrix = build_distance_matrix(num_cities, distances, city_format)
    except Exception as e:
        print(f"Error loading city file: {e}")
        return float('inf'), float('inf')
    
    # We'll collect multiple runs to account for randomness
    results = []
    
    for trial in range(num_trials):
        # ACO Core Algorithm based on AlgAenhanced.py
        
        # Calculate visibility matrix with safety check
        visibility = [[0 if i == j else 1.0 / (dist_matrix[i][j] if dist_matrix[i][j] != 0 else 1e-9)
                      for j in range(num_cities)] for i in range(num_cities)]
        
        # Initial tour using NN heuristic for tau bounds
        seed_tour = nearest_neighbour(dist_matrix)
        seed_length = calculate_tour_length(seed_tour, dist_matrix)
        
        # Setup pheromone matrix with MMAS bounds
        tau_max = 1.0 / (rho * seed_length) if rho * seed_length > 0 else 1.0
        tau_min = tau_max / (2.0 * num_cities) if num_cities > 0 else tau_max / 2.0
        tau_min = max(tau_min, 1e-9)
        
        pheromone = [[tau_max for _ in range(num_cities)] for _ in range(num_cities)]
        
        # Precompute candidate lists
        candidate_lists = [[] for _ in range(num_cities)]
        actual_candidate_size = min(candidate_list_size, num_cities - 1) if num_cities > 1 else 0
        if actual_candidate_size > 0:
            for i in range(num_cities):
                distances_from_i = []
                for j in range(num_cities):
                    if i == j: continue
                    if 0 <= i < num_cities and 0 <= j < num_cities:
                        distances_from_i.append((dist_matrix[i][j], j))
                distances_from_i.sort()
                candidate_lists[i] = [neighbor_idx for dist, neighbor_idx in distances_from_i[:actual_candidate_size]]
        
        # Run MMAS algorithm with the given parameters
        best_length = run_single_aco_trial(alpha, beta, pheromone, dist_matrix, visibility,
                                           num_cities, max_it, num_ants, rho, candidate_lists,
                                           actual_candidate_size, lambda_vis, tau_max, tau_min)
        
        results.append(best_length)
    
    # Return average tour length and best length across trials
    avg_length = sum(results) / len(results)
    best_length = min(results)
    
    print(f"Alpha={alpha:.2f}, Beta={beta:.2f}, Rho={rho:.2f}: Avg={avg_length:.2f}, Best={best_length}")
    return avg_length, best_length

def run_single_aco_trial(alpha, beta, pheromone, dist_matrix, visibility, 
                        num_cities, max_it, num_ants, rho, candidate_lists=None,
                        candidate_list_size=0, lambda_vis=0.5, tau_max=1.0, tau_min=1e-9):
    """
    Core MMAS algorithm implementation for a single trial, mimicking AlgAenhanced.py
    """
    # Already have visibility matrix passed in
    
    # Initialize best tour
    best_tour = None
    best_length = float('inf')
    
    # Track stagnation
    stagnation_counter = 0
    stagnation_limit = 50
    last_improvement_iteration = 0
    
    # Main MMAS loop
    for it in range(max_it):
        # Track iteration best
        iteration_best_tour = None
        iteration_best_length = float('inf')
        
        # Ant tours construction
        for ant in range(num_ants):
            # Use candidate lists and adaptive visibility for tour construction
            tour = construct_tour_enhanced(pheromone, visibility, alpha, beta, dist_matrix, 
                                         num_cities, candidate_lists, candidate_list_size, lambda_vis)
            
            # Calculate tour length
            tour_length = calculate_tour_length(tour, dist_matrix)
            
            # Update iteration best tour
            if tour_length < iteration_best_length:
                iteration_best_length = tour_length
                iteration_best_tour = tour.copy()
        
        # Apply 2-opt to the iteration's best tour
        if iteration_best_tour is not None:
            optimized_tour, optimized_length = two_opt(iteration_best_tour, dist_matrix)
            
            # Update global best if improved
            if optimized_length < best_length:
                best_length = optimized_length
                best_tour = optimized_tour.copy()
                stagnation_counter = 0
                last_improvement_iteration = it
            else:
                stagnation_counter += 1
        
        # Pheromone update (MMAS style)
        # Evaporation on all edges
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1 - rho)
                # Apply lower bound
                pheromone[i][j] = max(tau_min, pheromone[i][j])
        
        # Deposit from best tour (global best in this implementation)
        if best_tour is not None and best_length != float('inf') and best_length > 0:
            delta_tau = 1.0 / best_length
            for idx in range(num_cities):
                i = best_tour[idx]
                j = best_tour[(idx + 1) % num_cities]
                if 0 <= i < num_cities and 0 <= j < num_cities:
                    pheromone[i][j] += delta_tau
                    # Apply upper bound
                    pheromone[i][j] = min(tau_max, pheromone[i][j])
                    # Symmetric TSP
                    pheromone[j][i] = pheromone[i][j]
        
        # Stagnation handling - reset pheromones if needed
        if stagnation_counter >= stagnation_limit:
            # Reset pheromones to tau_max
            for i in range(num_cities):
                for j in range(num_cities):
                    pheromone[i][j] = tau_max
            stagnation_counter = 0
    
    # Return the best length found
    return best_length

def construct_tour_enhanced(pheromone, visibility, alpha, beta, dist_matrix, 
                         num_cities, candidate_lists=None, candidate_list_size=0, lambda_vis=0.5):
    """
    Construct a single ant tour using enhanced MMAS rules with:
    - Candidate Lists
    - Adaptive Visibility (considering return to start city)
    - Standard AS/MMAS probability rule
    """
    start_city = random.randrange(num_cities) if num_cities > 0 else 0
    visited = [False] * num_cities
    visited[start_city] = True
    tour = [start_city]
    current_city = start_city
    
    while len(tour) < num_cities:
        # Determine candidate cities using candidate lists if available
        possible_next_cities = []
        use_full_list = True
        
        if candidate_lists and candidate_list_size > 0 and 0 <= current_city < num_cities:
            candidate_neighbors = candidate_lists[current_city]
            unvisited_candidates = [j for j in candidate_neighbors if 0 <= j < num_cities and not visited[j]]
            if unvisited_candidates:
                possible_next_cities = unvisited_candidates
                use_full_list = False
        
        if use_full_list:
            possible_next_cities = [j for j, is_visited in enumerate(visited) if not is_visited]
        
        if not possible_next_cities:
            break  # No unvisited cities left
        
        # Calculate probabilities with adaptive visibility
        probabilities = []
        total_prob = 0.0
        
        for j in possible_next_cities:
            if not (0 <= current_city < num_cities and 0 <= j < num_cities):
                continue
            
            # Use pheromone value
            tau = pheromone[current_city][j] ** alpha
            
            # Calculate adaptive visibility considering return to start
            try:
                dist_to_j = dist_matrix[current_city][j]
                dist_j_to_start = dist_matrix[j][start_city]
                
                # Avoid division by zero
                denominator = dist_to_j + lambda_vis * dist_j_to_start
                denominator = max(denominator, 1e-10)  # Safety
                
                eta = (1.0 / denominator) ** beta
            except (IndexError, ZeroDivisionError):
                eta = 1e-10  # Safety default
            
            prob_val = tau * eta
            probabilities.append((j, prob_val))
            total_prob += prob_val
        
        # Select next city using roulette wheel selection
        if total_prob <= 0 or not probabilities:
            # Random selection if probabilities are invalid
            next_city = random.choice(possible_next_cities)
        else:
            # Roulette wheel selection
            r = random.random() * total_prob
            cumulative_prob = 0.0
            next_city = possible_next_cities[0]  # Default
            
            for city, prob in probabilities:
                cumulative_prob += prob
                if cumulative_prob >= r:
                    next_city = city
                    break
        
        # Add selected city to tour
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city
    
    return tour

def calculate_tour_length(tour, dist_matrix):
    """Calculate the total length of a tour with safety checks"""
    length = 0
    num_cities = len(tour)
    
    if num_cities < 2:
        return 0  # Empty tour or single city has length 0
    
    try:
        for i in range(num_cities - 1):
            # Check indices are valid
            if 0 <= tour[i] < len(dist_matrix) and 0 <= tour[i+1] < len(dist_matrix[tour[i]]):
                length += dist_matrix[tour[i]][tour[i+1]]
            else:
                # Handle invalid indices gracefully
                print(f"Warning: Invalid city indices in tour: {tour[i]}, {tour[i+1]}")
                return float('inf')
        
        # Add return to starting city with safety check
        if 0 <= tour[-1] < len(dist_matrix) and 0 <= tour[0] < len(dist_matrix[tour[-1]]):
            length += dist_matrix[tour[-1]][tour[0]]
        else:
            # Handle invalid indices gracefully
            print(f"Warning: Invalid city indices in tour return: {tour[-1]}, {tour[0]}")
            return float('inf')
        
    except IndexError:
        # Handle any unexpected index errors
        print("Error: IndexError during tour length calculation")
        return float('inf')
    
    return length

def nearest_neighbour(dist_matrix):
    """Generate a tour using nearest neighbor heuristic with safety checks"""
    num_cities = len(dist_matrix)
    if num_cities == 0:
        return []
    
    start_city = random.randrange(num_cities)
    
    tour = [start_city]
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)
    
    current_city = start_city
    while unvisited:
        # Find next city with minimum distance, handling potential zero distances
        best_dist = float('inf')
        next_city = None
        
        for city in unvisited:
            # Check index bounds to avoid errors
            if 0 <= current_city < len(dist_matrix) and 0 <= city < len(dist_matrix[current_city]):
                dist = dist_matrix[current_city][city]
                # Handle zero distances
                if dist == 0:
                    dist = 1e-10  # Small value instead of zero
                
                if dist < best_dist:
                    best_dist = dist
                    next_city = city
        
        if next_city is None:
            # If we couldn't find a valid next city, pick randomly
            if unvisited:
                next_city = random.choice(list(unvisited))
            else:
                break
        
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    return tour

def two_opt(tour, dist_matrix):
    """
    Applies 2-opt local search to improve a tour
    Returns improved tour and its length
    """
    n = len(tour)
    if n < 4:  # Need at least 4 cities for 2-opt to work
        return tour, calculate_tour_length(tour, dist_matrix)
    
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour, dist_matrix)
    
    # Allow more iterations for better optimization quality
    max_iterations = 20  # Increased from 5 to get better quality solutions
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Consider all possible pairs of non-adjacent edges
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:  # Skip adjacent edges
                    continue
                
                # Calculate the change in tour length if we reverse the segment
                # Current edges: (i-1,i) and (j,j+1 or 0)
                # New edges after reversal: (i-1,j) and (i,j+1 or 0)
                
                old_length = dist_matrix[best_tour[i-1]][best_tour[i]] + dist_matrix[best_tour[j]][best_tour[(j+1) % n]]
                new_length = dist_matrix[best_tour[i-1]][best_tour[j]] + dist_matrix[best_tour[i]][best_tour[(j+1) % n]]
                
                if new_length < old_length:
                    # Reverse the segment from i to j
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    best_length = best_length - old_length + new_length
                    improved = True
                    break  # Start over with the new tour
            
            if improved:
                break
    
    # Final verification
    final_length = calculate_tour_length(best_tour, dist_matrix)
    
    return best_tour, final_length

def grid_search(city_file, alphas, betas, rhos=None, lambda_vis_values=None, cl_sizes=None, trials_per_combo=3):
    """
    Perform a grid search to find the best parameters for the ACO algorithm.
    
    Parameters:
    - city_file: The city file to use for testing
    - alphas: List of alpha values to test
    - betas: List of beta values to test
    - rhos: List of rho values to test (optional)
    - lambda_vis_values: List of lambda_vis values to test (optional)
    - cl_sizes: List of candidate list sizes to test (optional)
    - trials_per_combo: Number of trials per parameter combination
    
    Returns:
    - List of results sorted by performance
    """
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default values if not provided
    if rhos is None:
        rhos = [0.15]  # Default rho
    if lambda_vis_values is None:
        lambda_vis_values = [0.5]  # Default lambda_vis
    if cl_sizes is None:
        cl_sizes = [20]  # Default candidate list size
    
    # Open a CSV file to log results
    filename = f"aco_param_search_{city_file.replace('.txt', '')}_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Alpha", "Beta", "Rho", "Lambda_vis", "CL_Size", "Avg Length", "Best Length"])
        
        # Calculate total combinations for progress tracking
        total_combos = len(alphas) * len(betas) * len(rhos) * len(lambda_vis_values) * len(cl_sizes)
        combo_count = 0
        
        # Iterate through all parameter combinations
        for alpha in alphas:
            for beta in betas:
                for rho in rhos:
                    for lambda_vis in lambda_vis_values:
                        for cl_size in cl_sizes:
                            combo_count += 1
                            print(f"\nTesting combination {combo_count}/{total_combos}:")
                            
                            # Run ACO with current parameter set
                            avg_length, best_length = run_aco_with_params(
                                alpha, beta, city_file, rho, lambda_vis, cl_size, trials_per_combo)
                            
                            # Store results
                            result = (alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length)
                            results.append(result)
                            
                            # Log to CSV
                            csvwriter.writerow(result)
                            csvfile.flush()  # Ensure data is written immediately
        
    # Sort by average tour length
    results.sort(key=lambda x: x[5])  # Sort by avg_length
    
    print(f"\nResults saved to {filename}")
    print("\nTop 5 Parameter Combinations:")
    for i, (alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length) in enumerate(results[:5], 1):
        print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}, Rho={rho:.2f}, Lambda_vis={lambda_vis:.2f}, CL={cl_size}: "
              f"Avg={avg_length:.2f}, Best={best_length}")
    
    return results

def random_search(city_file, num_trials=50, alpha_range=(0.0, 5.0), beta_range=(0.0, 10.0),
                rho_range=(0.05, 0.3), lambda_vis_range=(0.0, 2.0), cl_size_range=(5, 40)):
    """
    Perform a random search to find good parameters for the ACO algorithm
    
    Parameters:
    - city_file: The city file to use for testing
    - num_trials: Number of random parameter combinations to try
    - alpha_range: (min, max) for alpha values
    - beta_range: (min, max) for beta values
    - rho_range: (min, max) for rho values
    - lambda_vis_range: (min, max) for lambda_vis values
    - cl_size_range: (min, max) for candidate list sizes
    
    Returns:
    - List of results sorted by performance
    """
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Open a CSV file to log results
    filename = f"aco_random_search_{city_file.replace('.txt', '')}_{timestamp}.csv"
    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Alpha", "Beta", "Rho", "Lambda_vis", "CL_Size", "Avg Length", "Best Length"])
        
        # Generate and try random parameter combinations
        for trial in range(num_trials):
            # Generate random parameter values within ranges
            alpha = random.uniform(*alpha_range)
            beta = random.uniform(*beta_range)
            rho = random.uniform(*rho_range)
            lambda_vis = random.uniform(*lambda_vis_range)
            cl_size = random.randint(*cl_size_range)
            
            print(f"\nRandom Trial {trial+1}/{num_trials}:")
            avg_length, best_length = run_aco_with_params(
                alpha, beta, city_file, rho, lambda_vis, cl_size, trials_per_combo=2)
            
            # Store results
            result = (alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length)
            results.append(result)
            
            # Log to CSV
            csvwriter.writerow(result)
            csvfile.flush()  # Ensure data is written immediately
    
    # Sort by average tour length
    results.sort(key=lambda x: x[5])  # Sort by avg_length
    
    print(f"\nResults saved to {filename}")
    print("\nTop 5 Parameter Combinations from Random Search:")
    for i, (alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length) in enumerate(results[:5], 1):
        print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}, Rho={rho:.2f}, Lambda_vis={lambda_vis:.2f}, CL={cl_size}: "
              f"Avg={avg_length:.2f}, Best={best_length}")
    
    return results

def sequential_tuning(city_file, initial_alpha=1.0, initial_beta=2.0, initial_rho=0.15, 
                       initial_lambda_vis=0.5, initial_cl_size=20, trials_per_step=3):
    """
    Perform sequential parameter tuning, focusing only on alpha and beta parameters.
    
    Parameters:
    - city_file: The city file to use for testing
    - initial_alpha, initial_beta: Starting alpha and beta values
    - initial_rho, initial_lambda_vis, initial_cl_size: Fixed parameter values
    - trials_per_step: Number of trials per parameter value
    
    Returns:
    - Best parameter values found
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"aco_sequential_search_{city_file.replace('.txt', '')}_{timestamp}.csv"
    
    with open(results_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Phase", "Parameter", "Value", "Avg Length", "Best Length"])
        
        # Initialize best values - only alpha and beta will be tuned
        best_alpha = initial_alpha
        best_beta = initial_beta
        # These remain fixed
        best_rho = initial_rho
        best_lambda_vis = initial_lambda_vis
        best_cl_size = initial_cl_size
        
        # Phase 1: Find best alpha with fixed other parameters
        print("\nPhase 1: Finding best alpha value...")
        # Test wider range of alpha values
        alpha_range = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
        
        alpha_results = []
        for alpha in alpha_range:
            avg_length, best_length = run_aco_with_params(
                alpha, best_beta, city_file, best_rho, best_lambda_vis, best_cl_size, trials_per_step)
            result = ("Alpha", alpha, avg_length, best_length)
            alpha_results.append((alpha, avg_length, best_length))
            csvwriter.writerow(["Phase 1"] + list(result))
            csvfile.flush()
        
        # Find best alpha
        alpha_results.sort(key=lambda x: x[1])  # Sort by average length
        best_alpha = alpha_results[0][0]
        print(f"\nBest alpha found: {best_alpha:.2f}")
        
        # Phase 2: Find best beta with fixed other parameters
        print("\nPhase 2: Finding best beta value...")
        # Test wider range of beta values
        beta_range = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        
        beta_results = []
        for beta in beta_range:
            avg_length, best_length = run_aco_with_params(
                best_alpha, beta, city_file, best_rho, best_lambda_vis, best_cl_size, trials_per_step)
            result = ("Beta", beta, avg_length, best_length)
            beta_results.append((beta, avg_length, best_length))
            csvwriter.writerow(["Phase 2"] + list(result))
            csvfile.flush()
        
        # Find best beta
        beta_results.sort(key=lambda x: x[1])  # Sort by average length
        best_beta = beta_results[0][0]
        print(f"\nBest beta found: {best_beta:.2f}")
        
        # Final phase: Fine tuning of alpha and beta with best other parameters
        print("\nFinal Phase: Fine-tuning alpha and beta...")
        # More precise fine-tuning around best values
        fine_alpha_range = [
            max(0.1, best_alpha - 0.4), 
            max(0.1, best_alpha - 0.2),
            best_alpha, 
            min(5.0, best_alpha + 0.2),
            min(5.0, best_alpha + 0.4)
        ]
        fine_beta_range = [
            max(0.1, best_beta - 0.4),
            max(0.1, best_beta - 0.2), 
            best_beta,
            min(10.0, best_beta + 0.2),
            min(10.0, best_beta + 0.4)
        ]
        
        fine_results = []
        for alpha, beta in itertools.product(fine_alpha_range, fine_beta_range):
            avg_length, best_length = run_aco_with_params(
                alpha, beta, city_file, best_rho, best_lambda_vis, int(best_cl_size), trials_per_step)
            result = (alpha, beta, best_rho, best_lambda_vis, best_cl_size, avg_length, best_length)
            fine_results.append(result)
            csvwriter.writerow(["Final"] + ["Alpha,Beta", f"{alpha},{beta}", avg_length, best_length])
            csvfile.flush()
        
        # Sort and output final results
        fine_results.sort(key=lambda x: x[5])  # Sort by average length
        
        print(f"\nResults saved to {results_file}")
        print("\nTop 5 Final Parameter Combinations:")
        for i, (alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length) in enumerate(fine_results[:5], 1):
            print(f"{i}. Alpha={alpha:.2f}, Beta={beta:.2f}, Rho={rho:.2f}, Lambda_vis={lambda_vis:.2f}, CL={cl_size}: "
                  f"Avg={avg_length:.2f}, Best={best_length}")
        
        # Return best parameter values
        best_result = fine_results[0]
        return best_result[0], best_result[1], best_result[2], best_result[3], best_result[4]

def read_file_into_string(input_file, ord_range):
    """Function from AlgAenhanced.py to read city files"""
    try:
        the_file = open(input_file, 'r') 
        current_char = the_file.read(1) 
        file_string = ""
        length = len(ord_range)
        while current_char != "":
            i = 0
            while i < length:
                if ord(current_char) >= ord_range[i][0] and ord(current_char) <= ord_range[i][1]:
                    file_string = file_string + current_char
                    i = length
                else:
                    i = i + 1
            current_char = the_file.read(1)
        the_file.close()
        return file_string
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def remove_all_spaces(the_string):
    """Function from AlgAenhanced.py to process city files"""
    length = len(the_string)
    new_string = ""
    for i in range(length):
        if the_string[i] != " ":
            new_string = new_string + the_string[i]
    return new_string

def integerize(the_string):
    """Function from AlgAenhanced.py to process city files"""
    length = len(the_string)
    stripped_string = "0"
    for i in range(0, length):
        if ord(the_string[i]) >= 48 and ord(the_string[i]) <= 57:
            stripped_string = stripped_string + the_string[i]
    resulting_int = int(stripped_string)
    return resulting_int

def convert_to_list_of_int(the_string):
    """Function from AlgAenhanced.py to process city files"""
    list_of_integers = []
    location = 0
    finished = False
    while finished == False:
        found_comma = the_string.find(',', location)
        if found_comma == -1:
            finished = True
        else:
            list_of_integers.append(integerize(the_string[location:found_comma]))
            location = found_comma + 1
            if the_string[location:location + 5] == "NOTE=":
                finished = True
    return list_of_integers

def build_distance_matrix(num_cities, distances, city_format):
    """Function from AlgAenhanced.py to build distance matrix"""
    dist_matrix = []
    i = 0
    if city_format == "full":
        for j in range(num_cities):
            row = []
            for k in range(0, num_cities):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    elif city_format == "upper_tri":
        for j in range(0, num_cities):
            row = []
            for k in range(j):
                row.append(0)
            for k in range(num_cities - j):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    else:
        for j in range(0, num_cities):
            row = []
            for k in range(j + 1):
                row.append(0)
            for k in range(0, num_cities - (j + 1)):
                row.append(distances[i])
                i = i + 1
            dist_matrix.append(row)
    if city_format == "upper_tri" or city_format == "strict_upper_tri":
        for i in range(0, num_cities):
            for j in range(0, num_cities):
                if i > j:
                    dist_matrix[i][j] = dist_matrix[j][i]
    return dist_matrix

def batch_processing_all_city_files(max_trials=2, overnight=True):
    """
    Process all city files in sequence using grid search.
    
    Parameters:
    - max_trials: Number of trials per parameter combination
    - overnight: If True, uses more conservative settings for longer runs
    """
    # List of all city files to process, ordered by size
    city_files = [
        "AISearchfile012.txt",  # 12 cities
        "AISearchfile017.txt",  # 17 cities
        "AISearchfile021.txt",  # 21 cities
        "AISearchfile026.txt",  # 26 cities
        "AISearchfile042.txt",  # 42 cities
        "AISearchfile048.txt",  # 48 cities
        "AISearchfile058.txt",  # 58 cities
        "AISearchfile175.txt",  # 175 cities
        "AISearchfile180.txt",  # 180 cities
        "AISearchfile535.txt",  # 535 cities
    ]
    
    # Settings for grid search
    if overnight:
        print("Using overnight batch processing settings")
        # Alpha and beta are the key parameters to optimize
        alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        betas = [1.0, 2.0, 3.0, 4.0, 5.0]
        rhos = [0.15]  # Fixed rho value (standard for MMAS)
        lambda_vis_values = [0.5]  # Fixed lambda value
        cl_sizes = [20]  # Fixed candidate list size
    
    # Create a summary file for overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"grid_search_summary_{timestamp}.csv"
    
    with open(summary_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["City File", "Best Alpha", "Best Beta", "Best Rho", 
                           "Best Lambda_vis", "Best CL_Size", "Best Length"])
        
        # Process each city file
        for city_file in city_files:
            print(f"\n{'='*80}")
            print(f"PROCESSING CITY FILE: {city_file}")
            print(f"{'='*80}")
            
            try:
                # Run grid search
                print(f"\nRunning Grid Search on {city_file}...")
                results = grid_search(city_file, alphas, betas, rhos, lambda_vis_values, cl_sizes, trials_per_combo=max_trials)
                
                # Get the best result
                if results:
                    best_result = results[0]  # First is best (already sorted)
                    alpha, beta, rho, lambda_vis, cl_size, avg_length, best_length = best_result
                    
                    # Save to summary
                    csvwriter.writerow([city_file, alpha, beta, rho, lambda_vis, cl_size, best_length])
                    csvfile.flush()
            
            except Exception as e:
                print(f"Error processing {city_file}: {e}")
                # Log the error but continue with next file
                csvwriter.writerow([city_file, "ERROR", str(e)])
                csvfile.flush()
    
    print(f"\nBatch processing complete. Summary saved to {summary_file}")


if __name__ == "__main__":
    print("ACO Parameter Search Tool")
    print("=========================")
    print("This tool helps find optimal alpha and beta values for ACO algorithm.")
    print("\nChoose run mode:")
    print("1. Single city file")
    print("2. Batch process all city files (overnight run)")
    
    mode = input("Enter your choice (1-2): ")
    
    if mode == "2":
        # Batch processing mode
        print("\nBatch Processing Mode Selected")
        print("This will process all city files in sequence using grid search.")
        
        # Get max trials setting to control search time
        try:
            max_trials = int(input("Enter maximum number of trials per parameter combination (1-5, default=2): "))
            if max_trials < 1 or max_trials > 5:
                print("Using default value of 2 trials for overnight run")
                max_trials = 2
        except:
            print("Using default value of 2 trials for overnight run")
            max_trials = 2
        
        # Confirm before starting (as this is a long process)
        confirm = input("\nThis will run for several hours testing all city files.\nDo you want to proceed? (y/n): ").lower()
        if confirm == 'y':
            # Start batch processing
            batch_processing_all_city_files(max_trials, overnight=True)
        else:
            print("Operation cancelled.")
    
    else:
        # Single city file mode (simpler interface, grid search only)
        if len(sys.argv) > 1:
            city_file = sys.argv[1]
        else:
            city_file = input("Enter city file name (default: AISearchfile180.txt): ") or "AISearchfile180.txt"
        
        print(f"\nParameter search for ACO on city file: {city_file}")
        
        # Get max trials setting to control search time
        try:
            max_trials = int(input("Enter maximum number of trials per parameter combination (1-10, default=3): "))
            if max_trials < 1 or max_trials > 10:
                print("Using default value of 3 trials")
                max_trials = 3
        except:
            print("Using default value of 3 trials")
            max_trials = 3
        
        # Grid search parameters
        print("\nGrid Search Parameters:")
        # Alpha values
        try:
            alpha_input = input("Enter alpha values separated by commas (default: 0.5,1.0,1.5,2.0,2.5,3.0): ")
            alphas = [float(x) for x in alpha_input.split(",") if x.strip()] if alpha_input else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        except:
            print("Using default alpha values")
            alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            
        # Beta values  
        try:
            beta_input = input("Enter beta values separated by commas (default: 1.0,2.0,3.0,4.0,5.0): ")
            betas = [float(x) for x in beta_input.split(",") if x.strip()] if beta_input else [1.0, 2.0, 3.0, 4.0, 5.0]
        except:
            print("Using default beta values")
            betas = [1.0, 2.0, 3.0, 4.0, 5.0]
            
        # Fixed parameters
        rhos = [0.15]
        lambda_vis_values = [0.5]
        cl_sizes = [20]
        
        # Confirm test parameters (this helps avoid mistakes)
        print(f"\nTesting {len(alphas)} alpha values × {len(betas)} beta values = {len(alphas)*len(betas)} combinations")
        print(f"Each combination will be tested {max_trials} times")
        print(f"Total executions: {len(alphas)*len(betas)*max_trials}")
        
        confirm = input("\nDo you want to proceed? (y/n): ").lower()
        if confirm == 'y':
            print(f"\nStarting grid search with parameter combinations...")
            grid_search(city_file, alphas, betas, rhos, lambda_vis_values, cl_sizes, trials_per_combo=max_trials)
        else:
            print("Operation cancelled.")