import os
import sys
import time
import random
import math
from datetime import datetime


############ START OF SECTOR 0 (IGNORE THIS COMMENT)
############
############ NOW PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS.
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############ BY 'DO NOT TOUCH' I REALLY MEAN THIS. EVEN CHANGING THE SYNTAX, BY
############ ADDING SPACES OR COMMENTS OR LINE RETURNS AND SO ON, COULD MEAN THAT
############ CODES MIGHT NOT RUN WHEN I RUN THEM! DO NOT TOUCH MY COMMENTS EITHER!
############

def read_file_into_string(input_file, ord_range):
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

def remove_all_spaces(the_string):
    length = len(the_string)
    new_string = ""
    for i in range(length):
        if the_string[i] != " ":
            new_string = new_string + the_string[i]
    return new_string

def integerize(the_string):
    length = len(the_string)
    stripped_string = "0"
    for i in range(0, length):
        if ord(the_string[i]) >= 48 and ord(the_string[i]) <= 57:
            stripped_string = stripped_string + the_string[i]
    if len(stripped_string) > 1:
        stripped_string = stripped_string[1:]
    return int(stripped_string)

def convert_to_list_of_int(the_string):
    list_of_integers = []
    location = 0
    finished = False
    while finished == False:
        found_comma = the_string.find(',', location)
        if found_comma == -1:
            finished = True
            location = len(the_string)
        else:
            list_of_integers.append(integerize(the_string[location:found_comma]))
            location = found_comma + 1
            if the_string[location:].startswith("NOTE="):
                finished = True
    return list_of_integers

def build_distance_matrix(num_cities, distances, city_format):
    dist_matrix = []
    i = 0
    for j in range(num_cities):
        row = []
        for k in range(num_cities):
            row.append(0)
        dist_matrix.append(row)
    
    if city_format == "full":
        for j in range(num_cities):
            for k in range(num_cities):
                dist_matrix[j][k] = distances[i]
                i = i + 1
    elif city_format == "upper_tri":
        for j in range(num_cities):
            for k in range(j, num_cities):
                dist_matrix[j][k] = distances[i]
                if j != k:
                    dist_matrix[k][j] = distances[i]
                i = i + 1
    elif city_format == "strict_upper_tri":
        for j in range(num_cities):
            for k in range(j+1, num_cities):
                dist_matrix[j][k] = distances[i]
                dist_matrix[k][j] = distances[i]
                i = i + 1
    
    return dist_matrix

def read_in_algorithm_codes_and_tariffs(alg_codes_file):
    flag = "good"
    code_dictionary = {}
    tariff_dictionary = {}
    if not os.path.exists(alg_codes_file):
        flag = "not_exist"
        return code_dictionary, tariff_dictionary, flag
    ord_range = [[32, 126]]
    file_string = read_file_into_string(alg_codes_file, ord_range)
    file_string = remove_all_spaces(file_string)
    location = 0
    EOF = False
    list_of_items = []
    while EOF == False:
        found_comma = file_string.find(",", location)
        if found_comma == -1:
            EOF = True
            sandwich = file_string[location:]
        else:
            sandwich = file_string[location:found_comma]
            location = found_comma + 1
        list_of_items.append(sandwich)
    third_length = int(len(list_of_items)/3)
    for i in range(third_length):
        code_dictionary[list_of_items[3*i]] = list_of_items[3*i+1]
        tariff_dictionary[list_of_items[3*i]] = int(list_of_items[3*i+2])
    return code_dictionary, tariff_dictionary, flag

############
############ HAVE YOU TOUCHED ANYTHING ABOVE? BECAUSE EVEN CHANGING ONE CHARACTER OR
############ ADDING ONE SPACE OR LINE RETURN WILL MEAN THAT THE PROGRAM YOU HAND IN
############ MIGHT NOT RUN PROPERLY!
############
############ THE RESERVED VARIABLE 'input_file' IS THE CITY FILE UNDER CONSIDERATION.
############
############ IT CAN BE SUPPLIED BY SETTING THE VARIABLE BELOW OR VIA A COMMAND-LINE
############ EXECUTION OF THE FORM 'python skeleton.py city_file.txt'. WHEN SUPPLYING
############ THE CITY FILE VIA A COMMAND-LINE EXECUTION, ANY ASSIGNMENT OF THE VARIABLE
############ 'input_file' IN THE LINE BELOW IS SUPPRESSED.
############
############ IT IS ASSUMED THAT THIS PROGRAM 'skeleton.py' SITS IN A FOLDER THE NAME OF
############ WHICH IS YOUR USER-NAME, IN LOWER CASE, E.G., 'abcd12', WHICH IN TURN SITS
############ IN ANOTHER FOLDER. IN THIS OTHER FOLDER IS THE FOLDER 'city-files' AND NO
############ MATTER HOW THE NAME OF THE CITY FILE IS SUPPLIED TO THIS PROGRAM, IT IS
############ ASSUMED THAT THE CITY FILE IS IN THE FOLDER 'city-files'.
############
############ END OF SECTOR 0 (IGNORE THIS COMMENT)

input_file = "AISearchfile175.txt"

############ START OF SECTOR 1 (IGNORE THIS COMMENT)
############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

if len(sys.argv) > 1:
    input_file = sys.argv[1]
    if os.path.isabs(input_file):
        # If it's an absolute path, take just the filename
        input_file = os.path.basename(input_file)

############ END OF SECTOR 1 (IGNORE THIS COMMENT)

path_for_city_files = "../city-files"
if not os.path.exists(path_for_city_files):a
path_for_city_files = "city-files"
path_to_city_file = path_for_city_files + "/" + input_file
if not os.path.exists(path_to_city_file):
    print("The city file '" + input_file + "' does not exist!")
    sys.exit()

############ START OF SECTOR 3 (IGNORE THIS COMMENT)
ord_range = [[32, 126]]
file_string = read_file_into_string(path_to_city_file, ord_range)
file_string = remove_all_spaces(file_string)
print("I have found and read the input file " + input_file + ":")

location = file_string.find("SIZE=")
if location == -1:
    print("Error: could not find 'SIZE='")
    sys.exit()
    
comma = file_string.find(",", location)
if comma == -1:
    print("Error: could not find comma after 'SIZE='")
    sys.exit()
    
num_cities_as_string = file_string[location + 5:comma]
num_cities = integerize(num_cities_as_string)
print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))

comma_plus_one = comma + 1
location = file_string.find("NOTE=", comma_plus_one)
if location == -1:
    location = len(file_string)
    
distances_as_string = file_string[comma_plus_one:location]
distances = convert_to_list_of_int(distances_as_string)

counted_distances = len(distances)
if counted_distances == num_cities * num_cities:
    city_format = "full"
elif counted_distances == (num_cities * (num_cities + 1))/2:
    city_format = "upper_tri"
elif counted_distances == (num_cities * (num_cities - 1))/2:
    city_format = "strict_upper_tri"
else:
    print("Error: There are " + str(counted_distances) + " distances. This is not consistent with num_cities = " + str(num_cities))
    sys.exit()
    
dist_matrix = build_distance_matrix(num_cities, distances, city_format)
print("   the distance matrix 'dist_matrix' has been built.")

############
############ HAVE YOU TOUCHED ANYTHING ABOVE? BECAUSE EVEN CHANGING ONE CHARACTER OR
############ ADDING ONE SPACE OR LINE RETURN WILL MEAN THAT THE PROGRAM YOU HAND IN
############ MIGHT NOT RUN PROPERLY!
############
############ YOU NOW HAVE THE NUMBER OF CITIES STORED IN THE INTEGER VARIABLE 'num_cities'
############ AND THE TWO_DIMENSIONAL MATRIX 'dist_matrix' HOLDS THE INTEGER CITY-TO-CITY
############ DISTANCES SO THAT 'dist_matrix[i][j]' IS THE DISTANCE FROM CITY 'i' TO CITY 'j'.
############ BOTH 'num_cities' AND 'dist_matrix' ARE RESERVED VARIABLES AND SHOULD FEED
############ INTO YOUR IMPLEMENTATIONS.
############
############ THERE NOW FOLLOWS CODE THAT READS THE ALGORITHM CODES AND TARIFFS FROM
############ THE TEXT-FILE 'alg_codes_and_tariffs.txt' INTO THE RESERVED DICTIONARIES
############ 'code_dictionary' AND 'tariff_dictionary'. DO NOT AMEND THIS CODE!
############ THE TEXT FILE 'alg_codes_and_tariffs.txt' SHOULD BE IN THE SAME FOLDER AS
############ THE FOLDER 'city-files' AND THE FOLDER WHOSE NAME IS YOUR USER-NAME.
############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############
############ END OF SECTOR 3 (IGNORE THIS COMMENT)

path_for_alg_codes_and_tariffs = "../alg_codes_and_tariffs.txt"
if not os.path.exists(path_for_alg_codes_and_tariffs):
    path_for_alg_codes_and_tariffs = "alg_codes_and_tariffs.txt"
if not os.path.exists(path_for_alg_codes_and_tariffs):
    print("Error: I cannot find 'alg_codes_and_tariffs.txt'")
    sys.exit()

code_dictionary, tariff_dictionary, flag = read_in_algorithm_codes_and_tariffs(path_for_alg_codes_and_tariffs)

if flag != "good":
    print("Error: Something went wrong when reading 'alg_codes_and_tariffs.txt'")
    sys.exit()

print("The codes and tariffs have been read from 'alg_codes_and_tariffs.txt':")

############
############ HAVE YOU TOUCHED ANYTHING ABOVE? BECAUSE EVEN CHANGING ONE CHARACTER OR
############ ADDING ONE SPACE OR LINE RETURN WILL MEAN THAT THE PROGRAM YOU HAND IN
############ MIGHT NOT RUN PROPERLY! SORRY TO GO ON ABOUT THIS BUT YOU NEED TO BE
############ AWARE OF THIS FACT!
############
############ YOU NOW NEED TO SUPPLY SOME PARAMETERS.
############
############ THE RESERVED STRING VARIABLE 'my_user_name' SHOULD BE SET AT YOUR
############ USER-NAME, E.G., "abcd12"
############
############ END OF SECTOR 5 (IGNORE THIS COMMENT)

my_user_name = "fnlz75"

############
############ YOU CAN SUPPLY, IF YOU WANT, YOUR FULL NAME. THIS IS NOT USED AT ALL BUT SERVES AS
############ AN EXTRA CHECK THAT THIS FILE BELONGS TO YOU. IF YOU DO NOT WANT TO SUPPLY YOUR
############ NAME THEN EITHER SET THE STRING VARIABLES 'my_first_name' AND 'my_last_name' AT
############ SOMETHING LIKE "Mickey" AND "Mouse" OR AS THE EMPTY STRING (AS THEY ARE NOW;
############ BUT PLEASE ENSURE THAT THE RESERVED VARIABLES 'my_first_name' AND 'my_last_name'
############ ARE SET AT SOMETHING).
############
############ END OF SECTOR 6 (IGNORE THIS COMMENT)

my_first_name = ""
my_last_name = ""

############
############ YOU NEED TO SUPPLY THE ALGORITHM CODE IN THE RESERVED STRING VARIABLE 'algorithm_code'
############ FOR THE ALGORITHM YOU ARE IMPLEMENTING. IT NEEDS TO BE A LEGAL CODE FROM THE TEXT-FILE
############ 'alg_codes_and_tariffs.txt' (READ THIS FILE TO SEE THE CODES).
############
############ END OF SECTOR 7 (IGNORE THIS COMMENT)

algorithm_code = "PS"  # Particle Swarm Optimization with MMAS

############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

if not algorithm_code in code_dictionary:
    print("Error: the algorithm code " + algorithm_code + " is not in 'alg_codes_and_tariffs.txt'")
    sys.exit()
print("   your algorithm code is legal and is " + algorithm_code + " -" + code_dictionary[algorithm_code] + ".")

############
############ HAVE YOU TOUCHED ANYTHING ABOVE? BECAUSE EVEN CHANGING ONE CHARACTER OR
############ ADDING ONE SPACE OR LINE RETURN WILL MEAN THAT THE PROGRAM YOU HAND IN
############ MIGHT NOT RUN PROPERLY! SORRY TO GO ON ABOUT THIS BUT YOU NEED TO BE
############ AWARE OF THIS FACT!
############
############ NOW THE RESERVED STRING VARIABLE 'file_name' IS USED TO DEFINE THE NAME OF THE
############ OUTPUT TOUR FILE THAT WILL BE CREATED. DO NOT CHANGE THIS!
############

file_name = my_user_name + "_" + algorithm_code + ".txt"

############
############ THE RESERVED INTEGER VARIABLE 'my_seed' GIVES THE RANDOM SEED THAT WILL BE USED IN
############ ALL STOCHASTIC COMPONENTS OF YOUR ALGORITHM. DO NOT CHANGE THIS!
############

my_seed = 1000

############
############ YOU CAN ADD A NOTE THAT WILL BE ADDED AT THE END OF THE RESULTING TOUR FILE IF YOU LIKE,
############ E.G., "in my basic greedy search, I broke ties by always visiting the first
############ city found" BY USING THE RESERVED STRING VARIABLE 'added_note' OR LEAVE IT EMPTY
############ IF YOU WISH. THIS HAS NO EFFECT ON MARKS BUT HELPS YOU TO REMEMBER THINGS ABOUT
############ YOUR TOUR THAT YOU MIGHT BE INTERESTED IN LATER.
############
############ END OF SECTOR 8 (IGNORE THIS COMMENT)

added_note = "Advanced Hybrid PSO-MMAS: Combines PSO for global exploration with MMAS for local optimization. Enhanced with 3-opt search, adaptive parameters, diversification strategies, and Variable Neighborhood Search for final optimization. Prevents premature termination through advanced stagnation management."

############
############ NOW YOUR CODE SHOULD BEGIN.
############

# Initialize random seed for reproducibility
random.seed(my_seed)

# === Helper Functions ===

def calculate_tour_length(tour):
    """Calculate the total length of a tour"""
    length = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]  # Wrap around to complete the tour
        length += dist_matrix[from_city][to_city]
    return length

def generate_random_tour():
    """Generate a random tour (permutation of cities)"""
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

def generate_nearest_neighbor_tour(start_city=None):
    """Generate a tour using the nearest neighbor heuristic"""
    if start_city is None:
        start_city = random.randint(0, num_cities - 1)
    
    unvisited = set(range(num_cities))
    tour = [start_city]
    unvisited.remove(start_city)
    
    while unvisited:
        current_city = tour[-1]
        next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
        tour.append(next_city)
        unvisited.remove(next_city)
    
    return tour

def apply_2opt_local_search(tour, max_iterations=100):
    """Apply 2-opt local search to improve a tour"""
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # For larger problems, use random sampling for efficiency
        i_range = list(range(1, num_cities - 1))
        j_range = list(range(2, num_cities))
        
        if num_cities > 100:
            # Sample a subset of positions for large problems
            sample_size = min(num_cities // 2, 50)
            i_range = random.sample(i_range, sample_size)
            j_range = random.sample(j_range, sample_size)
        
        for i in i_range:
            for j in j_range:
                if j <= i:
                    continue  # Skip invalid 2-opt pairs
                
                # Cities involved in the potential swap
                a, b = best_tour[i-1], best_tour[i]
                c, d = best_tour[j], best_tour[(j+1) % num_cities]
                
                # Calculate change in tour length if we reverse segment from i to j
                current_length = dist_matrix[a][b] + dist_matrix[c][d]
                new_length = dist_matrix[a][c] + dist_matrix[b][d]
                delta = new_length - current_length
                
                if delta < 0:  # If the new tour would be shorter
                    # Reverse the segment from i to j
                    best_tour[i:j+1] = list(reversed(best_tour[i:j+1]))
                    best_length += delta
                    improved = True
                    break  # Start again with the new tour
            
            if improved:
                break
    
    return best_tour, calculate_tour_length(best_tour)  # Return the exact length for safety

def apply_3opt_local_search(tour, max_iterations=100, sample_size=None):
    """Apply 3-opt local search to improve a tour
    
    3-opt considers removing 3 edges and reconnecting in a different way (there are 8 possible ways 
    to reconnect, but only 4 that maintain a valid tour). This is more powerful than 2-opt.
    """
    best_tour = tour.copy()
    best_length = calculate_tour_length(best_tour)
    
    # Determine sample size based on problem size
    if sample_size is None:
        if num_cities <= 100:
            sample_size = num_cities  # Full search for small problems
        else:
            sample_size = min(num_cities // 2, 100)  # Sample for large problems
    
    iteration = 0
    improved = True
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Consider all possible triplets of edges or sample them
        edge_indices = list(range(num_cities))
        
        if len(edge_indices) > sample_size:
            # Sample edge indices for efficiency
            edge_indices = sorted(random.sample(edge_indices, sample_size))
        
        # Try all combinations of 3 edges
        for i_idx in range(len(edge_indices) - 2):
            i = edge_indices[i_idx]
            for j_idx in range(i_idx + 1, len(edge_indices) - 1):
                j = edge_indices[j_idx]
                for k_idx in range(j_idx + 1, len(edge_indices)):
                    k = edge_indices[k_idx]
                    
                    # Ensure i < j < k
                    if not (i < j < k) or j == i + 1 or k == j + 1:
                        continue
                    
                    # Get city indices (wrap around for the tour)
                    a = best_tour[i]
                    b = best_tour[(i + 1) % num_cities]
                    c = best_tour[j]
                    d = best_tour[(j + 1) % num_cities]
                    e = best_tour[k]
                    f = best_tour[(k + 1) % num_cities]
                    
                    # Current configuration: a-b, c-d, e-f
                    current_distance = (
                        dist_matrix[a][b] + 
                        dist_matrix[c][d] + 
                        dist_matrix[e][f]
                    )
                    
                    # Try all possible 3-opt reconnection configurations (4 valid ones)
                    # Option 1: a-c, b-e, d-f (reverse segment from b to c and d to e)
                    opt1_distance = (
                        dist_matrix[a][c] + 
                        dist_matrix[b][e] + 
                        dist_matrix[d][f]
                    )
                    
                    # Option 2: a-d, e-c, b-f (reverse segment from b to d and c to e)
                    opt2_distance = (
                        dist_matrix[a][d] + 
                        dist_matrix[e][c] + 
                        dist_matrix[b][f]
                    )
                    
                    # Option 3: a-e, d-b, c-f (reverse segment from b to c and d to e)
                    opt3_distance = (
                        dist_matrix[a][e] + 
                        dist_matrix[d][b] + 
                        dist_matrix[c][f]
                    )
                    
                    best_opt = -1
                    best_delta = 0
                    
                    if opt1_distance < current_distance:
                        best_delta = opt1_distance - current_distance
                        best_opt = 1
                    
                    if opt2_distance < current_distance and (best_opt == -1 or opt2_distance < opt1_distance):
                        best_delta = opt2_distance - current_distance
                        best_opt = 2
                    
                    if opt3_distance < current_distance and (best_opt == -1 or opt3_distance < min(opt1_distance, opt2_distance)):
                        best_delta = opt3_distance - current_distance
                        best_opt = 3
                    
                    if best_opt != -1:
                        # Apply the best 3-opt move
                        new_tour = best_tour.copy()
                        
                        if best_opt == 1:
                            # a-c, b-e, d-f (reverse segment from b to c and d to e)
                            new_tour[(i+1) % num_cities:(j+1) % num_cities] = list(reversed(new_tour[(i+1) % num_cities:(j+1) % num_cities]))
                            new_tour[(j+1) % num_cities:(k+1) % num_cities] = list(reversed(new_tour[(j+1) % num_cities:(k+1) % num_cities]))
                        elif best_opt == 2:
                            # a-d, e-c, b-f (reverse segment from b to d and c to e)
                            segment1 = new_tour[(i+1) % num_cities:(j+1) % num_cities]
                            segment2 = new_tour[(j+1) % num_cities:(k+1) % num_cities]
                            new_tour[(i+1) % num_cities:(i+1) % num_cities + len(segment2)] = segment2
                            new_tour[(i+1) % num_cities + len(segment2):(i+1) % num_cities + len(segment2) + len(segment1)] = segment1
                        elif best_opt == 3:
                            # a-e, d-b, c-f (reverse segment from b to c and d to e)
                            segment1 = list(reversed(new_tour[(i+1) % num_cities:(j+1) % num_cities]))
                            segment2 = list(reversed(new_tour[(j+1) % num_cities:(k+1) % num_cities]))
                            new_tour[(i+1) % num_cities:(i+1) % num_cities + len(segment2)] = segment2
                            new_tour[(i+1) % num_cities + len(segment2):(i+1) % num_cities + len(segment2) + len(segment1)] = segment1
                        
                        # Check if the move actually improved the tour
                        new_length = calculate_tour_length(new_tour)
                        if new_length < best_length:
                            best_tour = new_tour
                            best_length = new_length
                            improved = True
                            break
                
                if improved:
                    break
            
            if improved:
                break
    
    # Final verification of tour length
    final_length = calculate_tour_length(best_tour)
    return best_tour, final_length

# === PSO Functions ===

def get_swap_sequence(source, target):
    """Generate a sequence of swaps to transform source into target"""
    swaps = []
    source_copy = source.copy()
    
    for i in range(len(source)):
        if source_copy[i] != target[i]:
            j = source_copy.index(target[i], i)
            source_copy[i], source_copy[j] = source_copy[j], source_copy[i]
            swaps.append((i, j))
    
    return swaps

def apply_swaps(tour, swaps):
    """Apply a sequence of swaps to a tour"""
    new_tour = tour.copy()
    for i, j in swaps:
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def order_crossover(parent1, parent2):
    """Order Crossover (OX) operator for permutation problems"""
    n = len(parent1)
    
    # Select a random segment
    a, b = sorted(random.sample(range(n), 2))
    
    # Initialize child with placeholder values (-1)
    child = [-1] * n
    
    # Copy the segment from parent1 to child
    for i in range(a, b + 1):
        child[i] = parent1[i]
    
    # Track the cities already in the child
    used_cities = set(child[a:b+1])
    
    # Fill the remaining positions with cities from parent2 in order
    j = 0
    for i in range(n):
        if child[i] == -1:  # If position needs to be filled
            # Find the next city from parent2 that's not already in the child
            while parent2[j] in used_cities:
                j = (j + 1) % n
            
            child[i] = parent2[j]
            used_cities.add(parent2[j])
            j = (j + 1) % n
    
    return child

# === MMAS Functions ===

def initialize_pheromones(init_value=None):
    """Initialize the pheromone matrix with a uniform value"""
    if init_value is None:
        # Default initialization using heuristic information
        # Find a good initial tour using nearest neighbor + 2-opt
        nn_tour = generate_nearest_neighbor_tour()
        improved_tour, tour_length = apply_2opt_local_search(nn_tour)
        # Set initial pheromone proportional to the tour length
        init_value = 1.0 / tour_length
    
    # Initialize pheromone matrix
    pheromones = [[init_value for _ in range(num_cities)] for _ in range(num_cities)]
    return pheromones

def update_pheromones_mmas(pheromones, tours, tour_lengths, global_best_tour, global_best_length, evaporation_rate, iteration, max_iterations):
    """Update pheromones using the Max-Min Ant System (MMAS) approach
    
    MMAS key features:
    1. Only the best ant updates pheromones (iteration best, global best, or both)
    2. Pheromone values are bounded by max and min limits
    3. Pheromone trails are initialized to the max limit
    4. Stagnation detection and management
    """
    # Find the iteration's best tour
    iter_best_idx = tour_lengths.index(min(tour_lengths))
    iter_best_tour = tours[iter_best_idx]
    iter_best_length = tour_lengths[iter_best_idx]
    
    # Determine which tour to use for update based on iteration progress
    # Early iterations: use iteration best more often
    # Later iterations: use global best more often
    use_global_best_probability = 0.2 + 0.8 * (iteration / max_iterations)
    
    if random.random() < use_global_best_probability and global_best_length <= iter_best_length:
        # Use global best tour
        update_tour = global_best_tour
        update_length = global_best_length
    else:
        # Use iteration best tour
        update_tour = iter_best_tour
        update_length = iter_best_length
    
    # Calculate pheromone deposit amount
    deposit_amount = 1.0 / update_length
    
    # MMAS pheromone bounds calculation
    # Theoretical maximum based on optimal solution and evaporation rate
    p_max = 1.0 / (evaporation_rate * update_length)
    
    # Minimum pheromone value derived as suggested in MMAS papers
    # This ensures that the probability of constructing any tour remains non-zero
    avg_branching_factor = num_cities / 2  # Approximation
    p_best = 0.05  # Probability of constructing the best tour
    n_root = math.pow(1.0 - p_best, 1.0 / avg_branching_factor)
    p_min = p_max * (1.0 - n_root) / ((avg_branching_factor - 1.0) * n_root)
    
    # Evaporate all pheromones first
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                pheromones[i][j] *= (1.0 - evaporation_rate)
    
    # Deposit pheromones only on the selected best tour's edges
    for i in range(num_cities):
        from_city = update_tour[i]
        to_city = update_tour[(i + 1) % num_cities]
        
        # MMAS uses additive update
        pheromones[from_city][to_city] += deposit_amount
        pheromones[to_city][from_city] += deposit_amount  # Symmetric problem
    
    # Enforce min/max limits on all pheromone values
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                pheromones[i][j] = min(p_max, max(p_min, pheromones[i][j]))
    
    # Stagnation detection and management
    # If too many iterations without improvement, perform pheromone smoothing
    stagnation_threshold = max(20, num_cities // 3)
    
    if iteration > stagnation_threshold and iteration % stagnation_threshold == 0:
        # Smooth pheromones to encourage exploration
        smooth_factor = 0.5  # Blend factor 
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    # Blend current value with the average of min and max
                    avg_value = (p_max + p_min) / 2
                    pheromones[i][j] = (1.0 - smooth_factor) * pheromones[i][j] + smooth_factor * avg_value
    
    return pheromones

def construct_aco_tour(pheromones, alpha, beta):
    """Construct a tour using the ACO algorithm with pheromone and heuristic information"""
    tour = []
    unvisited = set(range(num_cities))
    
    # Start from a random city
    current_city = random.randint(0, num_cities - 1)
    tour.append(current_city)
    unvisited.remove(current_city)
    
    # Construct the tour
    while unvisited:
        # Calculate probabilities for next city
        probabilities = []
        
        for next_city in unvisited:
            # Pheromone value
            tau = pheromones[current_city][next_city]
            # Heuristic value (inverse of distance)
            eta = 1.0 / max(1, dist_matrix[current_city][next_city])  # Avoid division by zero
            
            # Calculate probability factor
            probability = (tau ** alpha) * (eta ** beta)
            probabilities.append((next_city, probability))
        
        # Sort by probability (descending)
        probabilities.sort(key=lambda x: x[1], reverse=True)
        
        # Roulette wheel selection
        total = sum(prob for _, prob in probabilities)
        if total == 0:  # Avoid division by zero
            # If all probabilities are zero, choose randomly
            next_city = random.choice(list(unvisited))
        else:
            # Use probabilities for weighted selection
            r = random.random() * total
            cum_prob = 0
            for city, prob in probabilities:
                cum_prob += prob
                if r <= cum_prob:
                    next_city = city
                    break
            else:
                # Fallback if something goes wrong
                next_city = probabilities[0][0]
        
        # Move to the next city
        tour.append(next_city)
        current_city = next_city
        unvisited.remove(next_city)
    
    return tour

# === Hybrid PSO-MMAS Algorithm ===

def run_pso_mmas_hybrid():
    """Run the hybrid PSO-MMAS algorithm"""
    print("\nStarting Hybrid PSO-MMAS Algorithm...")
    
    # === Initialization Phase ===
    # Scale parameters based on problem size
    # For very large problems (175+ cities), prioritize PSO and reduce MMAS
    if num_cities >= 175:
        num_particles = min(80, max(40, num_cities // 2))
        max_pso_iterations = max(80, min(120, num_cities // 2))  # More PSO iterations
        max_aco_iterations = min(30, max(15, num_cities // 6))   # Fewer MMAS iterations
        print(f"Large problem detected (n={num_cities}): Optimizing algorithm parameters")
    # For large problems
    elif num_cities >= 100:
        num_particles = min(100, max(50, num_cities // 2))
        max_pso_iterations = min(100, max(40, num_cities // 2))
        max_aco_iterations = min(50, max(20, num_cities // 4))
    # For medium/small problems
    else:
        num_particles = min(100, max(50, num_cities))
        max_pso_iterations = min(150, max(50, num_cities))
        max_aco_iterations = min(100, max(30, num_cities // 2))
    
    # PSO parameters
    w = 0.7  # Inertia weight
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    # MMAS parameters - adjust based on problem size
    if num_cities >= 175:  # Very large problems
        num_ants = min(20, max(10, num_cities // 8))  # Fewer ants
        alpha = 1.0  # Pheromone influence
        beta = 6.0   # Increased heuristic influence for larger problems
        rho = 0.15   # Faster evaporation to focus search
    else:
        num_ants = min(30, max(10, num_cities // 5))
        alpha = 1.0  # Pheromone influence
        beta = 5.0   # Heuristic influence
        rho = 0.1    # Evaporation rate
    
    # Initialize time tracking
    start_time = time.time()
    pso_time = 0
    aco_time = 0
    
    # === Phase 1: PSO Global Search ===
    print(f"Phase 1: PSO Global Search with {num_particles} particles for {max_pso_iterations} iterations")
    pso_start_time = time.time()
    
    # Initialize particles with a diverse set of high-quality solutions
    particles = []
    
    # Particle initialization strategies
    init_strategies = {
        "nn_2opt": 0.3,      # 30% nearest neighbor + 2-opt
        "nn_3opt": 0.1,      # 10% nearest neighbor + 3-opt
        "greedy": 0.2,       # 20% greedy (multiple start nearest neighbor)
        "random_2opt": 0.2,  # 20% random + 2-opt
        "pure_random": 0.2   # 20% purely random
    }
    
    print("Creating diverse initial population...")
    
    # Track best initial solution
    best_initial_tour = None
    best_initial_length = float('inf')
    
    # Create particles based on different strategies
    strategy_counts = {s: int(p * num_particles) for s, p in init_strategies.items()}
    
    # Ensure we create exactly num_particles by adjusting the most frequent strategy
    total = sum(strategy_counts.values())
    if total < num_particles:
        max_strategy = max(init_strategies, key=init_strategies.get)
        strategy_counts[max_strategy] += (num_particles - total)
    elif total > num_particles:
        max_strategy = max(init_strategies, key=init_strategies.get)
        strategy_counts[max_strategy] -= (total - num_particles)
    
    particles_created = 0
    
    # Create particles using nearest neighbor + 2-opt
    for _ in range(strategy_counts["nn_2opt"]):
        start = random.randint(0, num_cities - 1)
        nn_tour = generate_nearest_neighbor_tour(start)
        improved_tour, improved_length = apply_2opt_local_search(nn_tour, max_iterations=200)
        
        particles.append({
            'position': improved_tour,
            'velocity': [],
            'pbest_position': improved_tour.copy(),
            'pbest_fitness': improved_length
        })
        
        if improved_length < best_initial_length:
            best_initial_tour = improved_tour.copy()
            best_initial_length = improved_length
        
        particles_created += 1
        if particles_created % 10 == 0:
            print(f"Created {particles_created}/{num_particles} particles...")
    
    # Create particles using nearest neighbor + 3-opt
    for _ in range(strategy_counts["nn_3opt"]):
        start = random.randint(0, num_cities - 1)
        nn_tour = generate_nearest_neighbor_tour(start)
        improved_tour, improved_length = apply_3opt_local_search(
            nn_tour, 
            max_iterations=100,
            sample_size=min(num_cities, 50)
        )
        
        particles.append({
            'position': improved_tour,
            'velocity': [],
            'pbest_position': improved_tour.copy(),
            'pbest_fitness': improved_length
        })
        
        if improved_length < best_initial_length:
            best_initial_tour = improved_tour.copy()
            best_initial_length = improved_length
        
        particles_created += 1
        if particles_created % 10 == 0:
            print(f"Created {particles_created}/{num_particles} particles...")
    
    # Create particles using greedy approach (multiple start nearest neighbor)
    for _ in range(strategy_counts["greedy"]):
        best_nn_tour = None
        best_nn_length = float('inf')
        
        # Try multiple starting cities
        for _ in range(3):  # Try 3 different starting points
            start = random.randint(0, num_cities - 1)
            tour = generate_nearest_neighbor_tour(start)
            length = calculate_tour_length(tour)
            
            if length < best_nn_length:
                best_nn_tour = tour
                best_nn_length = length
        
        particles.append({
            'position': best_nn_tour,
            'velocity': [],
            'pbest_position': best_nn_tour.copy(),
            'pbest_fitness': best_nn_length
        })
        
        if best_nn_length < best_initial_length:
            best_initial_tour = best_nn_tour.copy()
            best_initial_length = best_nn_length
        
        particles_created += 1
        if particles_created % 10 == 0:
            print(f"Created {particles_created}/{num_particles} particles...")
    
    # Create particles using random + 2-opt
    for _ in range(strategy_counts["random_2opt"]):
        random_tour = generate_random_tour()
        improved_tour, improved_length = apply_2opt_local_search(random_tour, max_iterations=100)
        
        particles.append({
            'position': improved_tour,
            'velocity': [],
            'pbest_position': improved_tour.copy(),
            'pbest_fitness': improved_length
        })
        
        if improved_length < best_initial_length:
            best_initial_tour = improved_tour.copy()
            best_initial_length = improved_length
        
        particles_created += 1
        if particles_created % 10 == 0:
            print(f"Created {particles_created}/{num_particles} particles...")
    
    # Create purely random particles
    for _ in range(strategy_counts["pure_random"]):
        tour = generate_random_tour()
        fitness = calculate_tour_length(tour)
        
        particles.append({
            'position': tour,
            'velocity': [],
            'pbest_position': tour.copy(),
            'pbest_fitness': fitness
        })
        
        particles_created += 1
        if particles_created % 10 == 0:
            print(f"Created {particles_created}/{num_particles} particles...")
    
    print(f"Created initial population with best tour length: {best_initial_length}")
    
    # Find initial global best
    gbest_idx = min(range(len(particles)), key=lambda i: particles[i]['pbest_fitness'])
    gbest_position = particles[gbest_idx]['pbest_position'].copy()
    gbest_fitness = particles[gbest_idx]['pbest_fitness']
    
    print(f"Initial best solution: {gbest_fitness}")
    
    # PSO Iterations
    iterations_without_improvement = 0
    max_stagnation = max(50, num_cities // 2)  # Increased stagnation threshold (was 20)
    
    for iteration in range(max_pso_iterations):
        # Track iteration improvement
        iteration_improved = False
        
        # Apply local search to global best more frequently when stagnating
        opt_application_frequency = max(1, 10 - (iterations_without_improvement // 10)) 
        if iteration % opt_application_frequency == 0:
            # Use 3-opt occasionally, especially when stagnating
            use_3opt = random.random() < (0.2 + min(0.5, iterations_without_improvement * 0.01))
            
            if use_3opt:
                improved_gbest, improved_fitness = apply_3opt_local_search(
                    gbest_position, 
                    max_iterations=50 + iterations_without_improvement // 2,
                    sample_size=min(num_cities, 50 + iterations_without_improvement)
                )
                if improved_fitness < gbest_fitness:
                    gbest_position = improved_gbest
                    gbest_fitness = improved_fitness
                    iteration_improved = True
                    print(f"Iteration {iteration + 1}: Global best improved with 3-opt to {gbest_fitness}")
            else:
                improved_gbest, improved_fitness = apply_2opt_local_search(
                    gbest_position, 
                    max_iterations=100 + iterations_without_improvement
                )
                if improved_fitness < gbest_fitness:
                    gbest_position = improved_gbest
                    gbest_fitness = improved_fitness
                    iteration_improved = True
                    print(f"Iteration {iteration + 1}: Global best improved with 2-opt to {gbest_fitness}")
        
        # Update each particle
        for i, particle in enumerate(particles):
            # Increase crossover probability when stagnating
            crossover_prob = 0.2 + min(0.4, iterations_without_improvement * 0.01)
            
            # Apply crossover with increasing probability during stagnation
            if random.random() < crossover_prob:
                # Crossover with global best
                new_position = order_crossover(particle['position'], gbest_position)
                new_fitness = calculate_tour_length(new_position)
                
                # Update position
                particle['position'] = new_position
                
                # Update personal best if improved
                if new_fitness < particle['pbest_fitness']:
                    particle['pbest_position'] = new_position.copy()
                    particle['pbest_fitness'] = new_fitness
                    
                    # Update global best if needed
                    if new_fitness < gbest_fitness:
                        gbest_position = new_position.copy()
                        gbest_fitness = new_fitness
                        iteration_improved = True
                        print(f"Iteration {iteration + 1}: New best tour = {gbest_fitness}")
            else:
                # Standard PSO update via swap sequences
                # Calculate velocity components
                cognitive_swaps = get_swap_sequence(particle['position'], particle['pbest_position'])
                social_swaps = get_swap_sequence(particle['position'], gbest_position)
                
                # Apply inertia: keep some previous velocity components
                new_velocity = []
                if particle['velocity']:
                    # Adaptive inertia weight - decreases when stagnating to increase exploration
                    adaptive_w = max(0.4, w - (iterations_without_improvement * 0.01))
                    inertia_count = int(adaptive_w * len(particle['velocity']))
                    if inertia_count > 0:
                        new_velocity.extend(random.sample(particle['velocity'], 
                                                         min(len(particle['velocity']), inertia_count)))
                
                # Apply cognitive component with increased influence during stagnation
                if cognitive_swaps:
                    adaptive_c1 = c1 + min(1.0, iterations_without_improvement * 0.02)  # Increase cognitive coefficient
                    cognitive_count = min(len(cognitive_swaps), 
                                         int(adaptive_c1 * random.random() * len(cognitive_swaps)))
                    if cognitive_count > 0:
                        new_velocity.extend(random.sample(cognitive_swaps, cognitive_count))
                
                # Apply social component with decreased influence during stagnation
                if social_swaps:
                    adaptive_c2 = max(0.5, c2 - (iterations_without_improvement * 0.01))  # Decrease social coefficient
                    social_count = min(len(social_swaps), 
                                      int(adaptive_c2 * random.random() * len(social_swaps)))
                    if social_count > 0:
                        new_velocity.extend(random.sample(social_swaps, social_count))
                
                # Update velocity and position
                particle['velocity'] = new_velocity
                new_position = apply_swaps(particle['position'], new_velocity)
                new_fitness = calculate_tour_length(new_position)
                
                # Update position
                particle['position'] = new_position
                
                # Update personal best if improved
                if new_fitness < particle['pbest_fitness']:
                    particle['pbest_position'] = new_position.copy()
                    particle['pbest_fitness'] = new_fitness
                    
                    # Update global best if needed
                    if new_fitness < gbest_fitness:
                        gbest_position = new_position.copy()
                        gbest_fitness = new_fitness
                        iteration_improved = True
                        print(f"Iteration {iteration + 1}: New best tour = {gbest_fitness}")
            
            # Apply 2-opt local search with increasing probability during stagnation
            local_search_prob = 0.05 + min(0.3, iterations_without_improvement * 0.01)
            if random.random() < local_search_prob:
                improved_position, improved_fitness = apply_2opt_local_search(
                    particle['position'],
                    max_iterations=50 + iterations_without_improvement // 2
                )
                
                if improved_fitness < particle['pbest_fitness']:
                    particle['position'] = improved_position
                    particle['pbest_position'] = improved_position.copy()
                    particle['pbest_fitness'] = improved_fitness
                    
                    # Update global best if needed
                    if improved_fitness < gbest_fitness:
                        gbest_position = improved_position.copy()
                        gbest_fitness = improved_fitness
                        iteration_improved = True
                        print(f"Iteration {iteration + 1}: New best tour with 2-opt = {gbest_fitness}")
        
        # Check for stagnation
        if iteration_improved:
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1
        
        # Print progress
        if (iteration + 1) % 10 == 0 or iteration == max_pso_iterations - 1:
            print(f"PSO Iteration {iteration + 1}/{max_pso_iterations}: Best = {gbest_fitness}, " +
                  f"No improvement: {iterations_without_improvement}/{max_stagnation}")
        
        # Instead of early termination, apply diversification when stagnating
        if iterations_without_improvement >= max_stagnation // 2 and iterations_without_improvement % 25 == 0:
            print(f"Stagnation detected at iteration {iteration + 1} - applying diversification")
            
            # Apply diversity-enhancing strategies:
            
            # 1. Reset a portion of the swarm with new random tours
            reset_count = num_particles // 4  # Reset 25% of particles
            reset_indices = random.sample(range(num_particles), reset_count)
            for idx in reset_indices:
                if random.random() < 0.7:  # 70% random tours
                    particles[idx]['position'] = generate_random_tour()
                else:  # 30% nearest neighbor tours
                    particles[idx]['position'] = generate_nearest_neighbor_tour()
                    
                particles[idx]['velocity'] = []
                particles[idx]['pbest_fitness'] = calculate_tour_length(particles[idx]['position'])
                particles[idx]['pbest_position'] = particles[idx]['position'].copy()
            
            # 2. Perturb the best solution
            perturbed_gbest = gbest_position.copy()
            # Apply random swaps
            for _ in range(num_cities // 10):
                i, j = random.sample(range(num_cities), 2)
                perturbed_gbest[i], perturbed_gbest[j] = perturbed_gbest[j], perturbed_gbest[i]
            
            # Apply 2-opt to the perturbed solution
            improved_perturbed, improved_fitness = apply_2opt_local_search(perturbed_gbest, max_iterations=200)
            
            if improved_fitness < gbest_fitness:
                gbest_position = improved_perturbed
                gbest_fitness = improved_fitness
                iteration_improved = True
                print(f"Diversification improved best solution to {gbest_fitness}")
            
            # Reset stagnation counter but not to zero
            iterations_without_improvement = max_stagnation // 4  # Partial reset
    
    pso_end_time = time.time()
    pso_time = pso_end_time - pso_start_time
    print(f"PSO phase completed in {pso_time:.2f} seconds")
    print(f"Best solution after PSO: {gbest_fitness}")
    
    # === Phase 2: MMAS Local Optimization ===
    print(f"\nPhase 2: MMAS Local Optimization with {num_ants} ants for {max_aco_iterations} iterations")
    aco_start_time = time.time()
    
    # Use the PSO global best tour to initialize pheromone trails
    best_tour = gbest_position
    best_tour_length = gbest_fitness
    
    # Initialize pheromones based on the best PSO solution
    tau0 = 1.0 / best_tour_length  # Initial pheromone value
    pheromones = initialize_pheromones(tau0)
    
    # Set initial pheromones higher on best tour edges
    for i in range(num_cities):
        from_city = best_tour[i]
        to_city = best_tour[(i + 1) % num_cities]
        pheromones[from_city][to_city] = tau0 * 2
        pheromones[to_city][from_city] = tau0 * 2  # Symmetric problem
    
    # Track MMAS improvements
    mmas_iterations_without_improvement = 0
    mmas_max_stagnation = max(50, num_cities // 2)  # Increased from 20
    
    # MMAS Iterations
    for iteration in range(max_aco_iterations):
        # Construct tours with all ants
        ant_tours = []
        ant_tour_lengths = []
        
        # Increase local search probability when stagnating
        local_search_prob = 0.3 + min(0.4, mmas_iterations_without_improvement * 0.01)
        
        for _ in range(num_ants):
            # Construct a new tour
            ant_tour = construct_aco_tour(pheromones, alpha, beta)
            
            # Apply local search with increasing probability during stagnation
            if random.random() < local_search_prob:
                # Use 3-opt occasionally, especially when stagnating
                use_3opt = random.random() < (0.15 + min(0.4, mmas_iterations_without_improvement * 0.01))
                
                if use_3opt:
                    # Apply 3-opt with adaptive iteration limit
                    max_3opt_iterations = 30 + mmas_iterations_without_improvement // 2
                    sample_size = min(num_cities, 40 + mmas_iterations_without_improvement)
                    
                    improved_tour, improved_length = apply_3opt_local_search(
                        ant_tour, 
                        max_iterations=max_3opt_iterations,
                        sample_size=sample_size
                    )
                else:
                    # Apply 2-opt with adaptive iteration limit
                    max_2opt_iterations = 50 + mmas_iterations_without_improvement // 2
                    improved_tour, improved_length = apply_2opt_local_search(
                        ant_tour, 
                        max_iterations=max_2opt_iterations
                    )
                
                ant_tour = improved_tour
                ant_tour_length = improved_length
            else:
                ant_tour_length = calculate_tour_length(ant_tour)
            
            ant_tours.append(ant_tour)
            ant_tour_lengths.append(ant_tour_length)
            
            # Update best solution if improved
            if ant_tour_length < best_tour_length:
                best_tour = ant_tour.copy()
                best_tour_length = ant_tour_length
                mmas_iterations_without_improvement = 0
                print(f"MMAS Iteration {iteration + 1}: New best tour = {best_tour_length}")
            
        # Update pheromones using MMAS approach
        pheromones = update_pheromones_mmas(
            pheromones, 
            ant_tours, 
            ant_tour_lengths, 
            best_tour,
            best_tour_length,
            rho, 
            iteration,
            max_aco_iterations
        )
        
        # Check for stagnation
        if min(ant_tour_lengths) >= best_tour_length:
            mmas_iterations_without_improvement += 1
        else:
            mmas_iterations_without_improvement = 0
        
        # Print progress
        if (iteration + 1) % 10 == 0 or iteration == max_aco_iterations - 1:
            print(f"MMAS Iteration {iteration + 1}/{max_aco_iterations}: Best = {best_tour_length}, " +
                  f"No improvement: {mmas_iterations_without_improvement}/{mmas_max_stagnation}")
        
        # Implement periodic pheromone resets and diversification to prevent stagnation
        if mmas_iterations_without_improvement > 0:
            # More frequent resets as stagnation increases
            if mmas_iterations_without_improvement % max(5, 20 - (mmas_iterations_without_improvement // 5)) == 0:
                print(f"Resetting some pheromone trails to encourage exploration...")
                
                # Stronger reset probability as stagnation increases
                reset_prob = 0.5 + min(0.3, mmas_iterations_without_improvement * 0.01)
                
                # Reinitialize some pheromone trails
                for i in range(num_cities):
                    for j in range(i+1, num_cities):
                        if random.random() < reset_prob:
                            pheromones[i][j] = tau0
                            pheromones[j][i] = tau0
                
                # Create a new set of diverse ants occasionally
                if mmas_iterations_without_improvement % 20 == 0:
                    print("Adding diversity with new seed tours...")
                    
                    # Add diversity with some nearest neighbor tours
                    for _ in range(num_ants // 5):
                        start = random.randint(0, num_cities - 1)
                        new_tour = generate_nearest_neighbor_tour(start)
                        improved_tour, improved_length = apply_2opt_local_search(new_tour, max_iterations=100)
                        
                        if improved_length < best_tour_length:
                            best_tour = improved_tour.copy()
                            best_tour_length = improved_length
                            mmas_iterations_without_improvement = 0
                            print(f"Diversification produced new best tour: {best_tour_length}")
                            break
    
    aco_end_time = time.time()
    aco_time = aco_end_time - aco_start_time
    print(f"MMAS phase completed in {aco_time:.2f} seconds")
    
    # Final optimization
    print("\nPerforming final multilevel optimization...")
    
    # Try multiple approaches for final optimization
    final_tour = best_tour.copy()
    final_length = best_tour_length
    
    # 1. Apply 3-opt with high iteration limit
    improved_tour, improved_length = apply_3opt_local_search(final_tour, max_iterations=200)
    if improved_length < final_length:
        print(f"3-opt improved solution from {final_length} to {improved_length}")
        final_tour = improved_tour
        final_length = improved_length
    
    # 2. Try a random restart with nearest neighbor + 3-opt
    for _ in range(3):  # Try 3 different starting points
        start = random.randint(0, num_cities - 1)
        nn_tour = generate_nearest_neighbor_tour(start)
        improved_tour, improved_length = apply_3opt_local_search(nn_tour, max_iterations=100)
        
        if improved_length < final_length:
            print(f"Random restart (NN + 3-opt) improved solution from {final_length} to {improved_length}")
            final_tour = improved_tour
            final_length = improved_length
    
    # 3. Try crossover between different good solutions
    # Maintain a pool of good solutions
    solution_pool = [final_tour]
    
    # Add some particles' personal bests to the pool if they're good enough
    good_threshold = final_length * 1.1  # Within 10% of best
    for particle in particles:
        if particle['pbest_fitness'] <= good_threshold:
            solution_pool.append(particle['pbest_position'])
    
    # Add some ant solutions to the pool
    for i, tour_length in enumerate(ant_tour_lengths):
        if tour_length <= good_threshold:
            solution_pool.append(ant_tours[i])
    
    # Try crossovers between solutions in the pool
    for _ in range(min(5, len(solution_pool))):
        if len(solution_pool) >= 2:
            parents = random.sample(solution_pool, 2)
            crossover_tour = order_crossover(parents[0], parents[1])
            improved_tour, improved_length = apply_2opt_local_search(crossover_tour, max_iterations=300)
            
            if improved_length < final_length:
                print(f"Crossover improved solution from {final_length} to {improved_length}")
                final_tour = improved_tour
                final_length = improved_length
                
                # Add this to the pool for further crossover
                solution_pool.append(final_tour)
    
    # 4. Apply Variable Neighborhood Search
    print("Applying Variable Neighborhood Search for final refinement...")
    current_tour = final_tour
    current_length = final_length
    
    # Try increasing levels of perturbation
    for level in range(1, 4):
        # Perturb the solution
        perturbed_tour = current_tour.copy()
        
        # Apply random swaps based on perturbation level
        num_swaps = level * (num_cities // 20)
        for _ in range(num_swaps):
            i, j = random.sample(range(num_cities), 2)
            perturbed_tour[i], perturbed_tour[j] = perturbed_tour[j], perturbed_tour[i]
        
        # Apply local search to perturbed solution
        if level == 1:
            # Light perturbation - use 2-opt
            improved_tour, improved_length = apply_2opt_local_search(perturbed_tour, max_iterations=500)
        else:
            # Stronger perturbation - use 3-opt
            improved_tour, improved_length = apply_3opt_local_search(perturbed_tour, max_iterations=100)
        
        if improved_length < current_length:
            print(f"VNS level {level} improved solution from {current_length} to {improved_length}")
            current_tour = improved_tour
            current_length = improved_length
            
            # Reset perturbation level when we find an improvement
            level = 0
    
    # 5. Apply final 3-opt with higher iteration limit to best solution
    if current_length < final_length:
        final_tour = current_tour
        final_length = current_length
    
    final_tour, final_length = apply_3opt_local_search(final_tour, max_iterations=300)
    
    # Update best tour if improved
    if final_length < best_tour_length:
        print(f"Final optimization improved tour from {best_tour_length} to {final_length}")
        best_tour = final_tour
        best_tour_length = final_length
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nPSO-MMAS Hybrid completed in {total_time:.2f} seconds")
    print(f"  - PSO phase: {pso_time:.2f} seconds ({pso_time/total_time*100:.1f}%)")
    print(f"  - MMAS phase: {aco_time:.2f} seconds ({aco_time/total_time*100:.1f}%)")
    print(f"Best tour length: {best_tour_length}")
    
    # Update note with timing information
    global added_note
    added_note += f"\nPSO phase: {pso_time:.2f}s, MMAS phase: {aco_time:.2f}s, Total: {total_time:.2f}s"
    
    return best_tour, best_tour_length

# Run the algorithm
start_time = time.time()
tour, tour_length = run_pso_mmas_hybrid()
end_time = time.time()

print(f"\nExecution completed in {end_time - start_time:.2f} seconds")
print(f"Best tour length found: {tour_length}")

############ END OF YOUR CODE
############ THE RESERVED VARIABLES 'tour' AND 'tour_length' NEED TO BE RETURN VALUES
############ FROM YOUR ALGORITHM. THEY ARE USED FOR SCORING YOUR TOUR. DO NOT REUSE THEM.
############ ENSURE YOU HAVEN'T MADE ANY ERRORS, ELSE YOUR TOUR WILL BE INVALID.

############
############ YOUR CODE SHOULD NOW BE COMPLETE AND WHEN EXECUTION OF THIS PROGRAM 'skeleton.py'
############ REACHES THIS POINT, YOU SHOULD HAVE COMPUTED A TOUR IN THE RESERVED VARIABLE 'tour',
############ WHICH HOLDS A LIST OF THE INTEGERS FROM {0, 1, ..., 'num_cities' - 1} GIVING THE ORDER
############ IN WHICH CITIES ARE VISITED IN THE TOUR, AND YOU SHOULD ALSO HOLD THE LENGTH OF THIS
############ TOUR IN THE RESERVED VARIABLE 'tour_length'.
############

# Check if tour exists
if not 'tour' in vars():
    tour = []
    for i in range(0, num_cities):
        tour.append(i)
    print("No tour defined: created a default tour.")
    print("The problem size was", num_cities, ".")

if not 'tour_length' in vars():
    tour_length = -1
    print("No tour_length defined: created a default tour_length.")

############
############ YOUR TOUR WILL BE PACKAGED IN A TOUR FILE OF THE APPROPRIATE FORMAT AND THIS TOUR FILE'S
############ NAME WILL BE A MIX OF THE NAME OF THE CITY FILE, THE NAME OF THIS PROGRAM AND THE
############ CURRENT DATE AND TIME. SO, EVERY SUCCESSFUL EXECUTION GIVES A TOUR FILE WITH A UNIQUE
############ NAME AND YOU CAN RENAME THE ONES YOU WANT TO KEEP LATER.
############

############
############ DO NOT TOUCH OR ALTER THE CODE BELOW THIS POINT! YOU HAVE BEEN WARNED!
############

flag = "good"
length = len(tour)
for i in range(0, length):
    if isinstance(tour[i], int) == False:
        flag = "bad"
    else:
        tour[i] = int(tour[i])
if flag == "bad":
    print("Error: Your tour contains non-integer values.")
    sys.exit()
if isinstance(tour_length, int) == False:
    print("Error: Your tour_length is a non-integer value.")
    sys.exit()
if len(tour) != num_cities:
    print("Error: Your tour has incorrect length.")
    sys.exit()
if sorted(tour) != list(range(num_cities)):
    print("Error: Your tour has incorrect cities.")
    sys.exit()
check_tour_length = 0
for i in range(0, num_cities):
    check_tour_length = check_tour_length + dist_matrix[tour[i]][tour[(i + 1) % num_cities]]
if check_tour_length != tour_length:
    flag = "bad"
if flag == "bad":
    print("Error: Your tour_length is not " + str(check_tour_length) + ".")
    sys.exit()
print("You, user " + my_user_name + ", have successfully built a tour of length " + str(tour_length) + "!")

local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]
output_file_name = script_name[0:len(script_name) - 3] + "_" + input_file[0:len(input_file) - 4] + "_" + algorithm_code + "_" + output_file_time + ".txt"

f = open(output_file_name,'w')
f.write("USER = " + my_user_name + " (" + my_first_name + " " + my_last_name + "),\n")
f.write("ALGORITHM CODE = " + algorithm_code + ", NAME OF CITY-FILE = " + input_file + ",\n")
f.write("SIZE = " + str(num_cities) + ", TOUR LENGTH = " + str(tour_length) + ",\n")
f.write(str(tour[0]))
for i in range(1,num_cities):
    f.write("," + str(tour[i]))
f.write(",\nNOTE = " + added_note + ".\n")
f.close()
print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")