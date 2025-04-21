############
############ ALTHOUGH I GIVE YOU THIS TEMPLATE PROGRAM WITH THE NAME 'skeleton.py', 
############ YOU CAN RENAME IT TO ANYTHING YOU LIKE. HOWEVER, FOR THE PURPOSES OF 
############ THE EXPLANATION IN THESE COMMENTS, I ASSUME THAT THIS PROGRAM IS STILL 
############ CALLED 'skeleton.py'.
############
############ IF YOU WISH TO IMPORT STANDARD MODULES, YOU CAN ADD THEM AFTER THOSE BELOW.
############ NOTE THAT YOU ARE NOT ALLOWED TO IMPORT ANY NON-STANDARD MODULES! TO SEE
############ THE STANDARD MODULES, TAKE A LOOK IN 'validate_before_handin.py'.
############
############ DO NOT INCLUDE ANY COMMENTS ON A LINE WHERE YOU IMPORT A MODULE.
############

import os
import sys
import time
import random
from datetime import datetime
import math

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
    resulting_int = int(stripped_string)
    return resulting_int

def convert_to_list_of_int(the_string):
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

def read_in_algorithm_codes_and_tariffs(alg_codes_file):
    flag = "good"
    code_dictionary = {}   
    tariff_dictionary = {}  
    if not os.path.exists(alg_codes_file):
        flag = "not_exist"  
        return code_dictionary, tariff_dictionary, flag
    ord_range = [[32, 126]]
    file_string = read_file_into_string(alg_codes_file, ord_range)  
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
        code_dictionary[list_of_items[3 * i]] = list_of_items[3 * i + 1]
        tariff_dictionary[list_of_items[3 * i]] = int(list_of_items[3 * i + 2])
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

input_file = "AISearchfile180.txt"

############ START OF SECTOR 1 (IGNORE THIS COMMENT)
############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

if len(sys.argv) > 1:
    input_file = sys.argv[1]

############ END OF SECTOR 1 (IGNORE THIS COMMENT)

############ START OF SECTOR 2 (IGNORE THIS COMMENT)
path_for_city_files = os.path.join("..", "city-files")
############ END OF SECTOR 2 (IGNORE THIS COMMENT)

############ START OF SECTOR 3 (IGNORE THIS COMMENT)
path_to_input_file = os.path.join(path_for_city_files, input_file)
if os.path.isfile(path_to_input_file):
    ord_range = [[32, 126]]
    file_string = read_file_into_string(path_to_input_file, ord_range)
    file_string = remove_all_spaces(file_string)
    print("I have found and read the input file " + input_file + ":")
else:
    print("*** error: The city file " + input_file + " does not exist in the city-file folder.")
    sys.exit()

location = file_string.find("SIZE=")
if location == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
comma = file_string.find(",", location)
if comma == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
    sys.exit()
    
num_cities_as_string = file_string[location + 5:comma]
num_cities = integerize(num_cities_as_string)
print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))

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
    print("*** error: The city file " + input_file + " is incorrectly formatted.")
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

############ START OF SECTOR 4 (IGNORE THIS COMMENT)
path_for_alg_codes_and_tariffs = os.path.join("..", "alg_codes_and_tariffs.txt")
############ END OF SECTOR 4 (IGNORE THIS COMMENT)

############ START OF SECTOR 5 (IGNORE THIS COMMENT)
code_dictionary, tariff_dictionary, flag = read_in_algorithm_codes_and_tariffs(path_for_alg_codes_and_tariffs)

if flag != "good":
    print("*** error: The text file 'alg_codes_and_tariffs.txt' does not exist.")
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

############ START OF SECTOR 6 (IGNORE THIS COMMENT)
############
############ YOU CAN SUPPLY, IF YOU WANT, YOUR FULL NAME. THIS IS NOT USED AT ALL BUT SERVES AS
############ AN EXTRA CHECK THAT THIS FILE BELONGS TO YOU. IF YOU DO NOT WANT TO SUPPLY YOUR
############ NAME THEN EITHER SET THE STRING VARIABLES 'my_first_name' AND 'my_last_name' AT 
############ SOMETHING LIKE "Mickey" AND "Mouse" OR AS THE EMPTY STRING (AS THEY ARE NOW;
############ BUT PLEASE ENSURE THAT THE RESERVED VARIABLES 'my_first_name' AND 'my_last_name'
############ ARE SET AT SOMETHING).
############
############ END OF SECTOR 6 (IGNORE THIS COMMENT)

my_first_name = "Morgan"
my_last_name = "Rosca"

############ START OF SECTOR 7 (IGNORE THIS COMMENT)
############
############ YOU NEED TO SUPPLY THE ALGORITHM CODE IN THE RESERVED STRING VARIABLE 'algorithm_code'
############ FOR THE ALGORITHM YOU ARE IMPLEMENTING. IT NEEDS TO BE A LEGAL CODE FROM THE TEXT-FILE
############ 'alg_codes_and_tariffs.txt' (READ THIS FILE TO SEE THE CODES).
############
############ END OF SECTOR 7 (IGNORE THIS COMMENT)

algorithm_code = "AC"

############ START OF SECTOR 8 (IGNORE THIS COMMENT)
############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

if not algorithm_code in code_dictionary:
    print("*** error: the algorithm code " + algorithm_code + " is illegal")
    sys.exit()
print("   your algorithm code is legal and is " + algorithm_code + " -" + code_dictionary[algorithm_code] + ".")

start_time = time.time()

############
############ HAVE YOU TOUCHED ANYTHING ABOVE? BECAUSE EVEN CHANGING ONE CHARACTER OR
############ ADDING ONE SPACE OR LINE RETURN WILL MEAN THAT THE PROGRAM YOU HAND IN
############ MIGHT NOT RUN PROPERLY! SORRY TO GO ON ABOUT THIS BUT YOU NEED TO BE 
############ AWARE OF THIS FACT!
############
############ YOU CAN ADD A NOTE THAT WILL BE ADDED AT THE END OF THE RESULTING TOUR FILE IF YOU LIKE,
############ E.G., "in my basic greedy search, I broke ties by always visiting the first 
############ city found" BY USING THE RESERVED STRING VARIABLE 'added_note' OR LEAVE IT EMPTY
############ IF YOU WISH. THIS HAS NO EFFECT ON MARKS BUT HELPS YOU TO REMEMBER THINGS ABOUT
############ YOUR TOUR THAT YOU MIGHT BE INTERESTED IN LATER. NOTE THAT I CALCULATE THE TIME OF
############ A RUN USING THE RESERVED VARIABLE 'start_time' AND INCLUDE THE RUN-TIME IN 'added_note' LATER.
############
############ IN FACT, YOU CAN INCLUDE YOUR ADDED NOTE IMMEDIATELY BELOW OR EVEN INCLUDE YOUR ADDED NOTE
############ AT ANY POINT IN YOUR PROGRAM: JUST DEFINE THE STRING VARIABLE 'added_note' WHEN YOU WISH
############ (BUT DON'T REMOVE THE ASSIGNMENT IMMEDIATELY BELOW).
############
############ END OF SECTOR 8 (IGNORE THIS COMMENT)

added_note = ""

############ START OF SECTOR 9 (IGNORE THIS COMMENT)
############
############ NOW YOUR CODE SHOULD BEGIN BUT FIRST A COMMENT.
############
############ IF YOU ARE IMPLEMENTING GA THEN:
############  - IF YOU EXECUTE YOUR MAIN LOOP A FIXED NUMBER OF TIMES THEN USE THE VARIABLE 'max_it' TO DENOTE THIS NUMBER
############  - USE THE VARIABLE 'pop_size' TO DENOTE THE SIZE OF YOUR POPULATION (THIS IS '|P|' IN THE PSEUDOCODE)
############
############ IF YOU ARE IMPLEMENTING AC THEN:
############  - IF YOU EXECUTE YOUR MAIN LOOP A FIXED NUMBER OF TIMES THEN USE THE VARIABLE 'max_it' TO DENOTE THIS NUMBER
############  - USE THE VARIABLE 'num_ants' TO DENOTE THE NUMBER OF ANTS (THIS IS 'N' IN THE PSEUDOCODE)
############
############ IF YOU ARE IMPLEMENTING PS THEN:
############  - IF YOU EXECUTE YOUR MAIN LOOP A FIXED NUMBER OF TIMES THEN USE THE VARIABLE 'max_it' TO DENOTE THIS NUMBER
############  - USE THE VARIABLE 'num_parts' TO DENOTE THE NUMBER OF PARTICLES (THIS IS 'N' IN THE PSEUDOCODE)
############
############ DOING THIS WILL MEAN THAT THIS INFORMATION IS WRITTEN WITHIN 'added_note' IN ANY TOUR-FILE PRODUCED.
############ OF COURSE, THE VALUES OF THESE VARIABLES NEED TO BE ACCESSIBLE TO THE MAIN BODY OF CODE.
############ IT'S FINE IF YOU DON'T ADOPT THESE VARIABLE NAMES BUT THIS USEFUL INFORMATION WILL THEN NOT BE WRITTEN TO ANY
############ TOUR-FILE PRODUCED BY THIS CODE.
############
############ END OF SECTOR 9 (IGNORE THIS COMMENT)

# ----- 2-opt Implementation -----
def calculate_tour_length(tour, dist_matrix):
    """Calculates the length of a given tour."""
    length = 0
    num_cities_in_tour = len(tour)
    if num_cities_in_tour < 2:
        return 0
    for i in range(num_cities_in_tour - 1):
        length += dist_matrix[tour[i]][tour[i + 1]]
    length += dist_matrix[tour[num_cities_in_tour - 1]][tour[0]] # Use num_cities_in_tour
    return length

def two_opt(tour, dist_matrix):
    """Improves a tour using the 2-opt heuristic."""
    num_cities_in_tour = len(tour)
    if num_cities_in_tour < 4:
        return tour, calculate_tour_length(tour, dist_matrix)

    best_tour = tour[:]
    best_length = calculate_tour_length(best_tour, dist_matrix)
    improvement_found = True
    
    # Limit 2-opt iterations for performance if needed, e.g., max_swaps = num_cities_in_tour * 5
    # swap_count = 0 

    while improvement_found: # and swap_count < max_swaps:
        improvement_found = False
        for i in range(num_cities_in_tour - 2):
            for j in range(i + 2, num_cities_in_tour):
                j_next = (j + 1) % num_cities_in_tour
                node_i, node_i_plus_1 = best_tour[i], best_tour[i+1]
                node_j, node_j_next = best_tour[j], best_tour[j_next]

                # Calculate distance change
                original_dist = dist_matrix[node_i][node_i_plus_1] + dist_matrix[node_j][node_j_next]
                new_dist = dist_matrix[node_i][node_j] + dist_matrix[node_i_plus_1][node_j_next]

                if new_dist < original_dist:
                    # Perform swap
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    best_length += (new_dist - original_dist)
                    improvement_found = True
                    # swap_count += 1
                    break # Exit inner loop (j) - first improvement
            if improvement_found:
                break # Exit outer loop (i) - restart search

    # Ensure integer length for consistency with assignment reqs
    return best_tour, int(round(best_length))


# ----- MMAS PARAMETERS -----
# Use num_cities from the outer scope
num_ants = 25

# Use num_ants calculated above
max_it = 200              # max iterations

alpha = 1.0                 # pheromone importance
beta  = 3.0                 # heuristic importance (visibility) - Adjusted
rho   = 0.8                 # pheromone evaporation rate (0<rho<=1) - Adjusted
# Q and elitist_weight are not used in this MMAS version

p_best = 0.05               # Parameter for tau_min heuristic calculation
tau_max = 0.0
tau_min = 1e-9              # Initialize with a very small positive value
stagnation_counter = 0
stagnation_limit = 50       # Iterations without improvement to trigger reinit
reset_pheromone_on_stagnation = True

# ---- helper structures ----
# Use dist_matrix from the outer scope
# visibility matrix η_{ij} = 1/d_{ij}
visibility = [[0 if i == j else 1.0 / (dist_matrix[i][j] if dist_matrix[i][j] != 0 else 1e-9)
               for j in range(num_cities)] for i in range(num_cities)]

# best tour trackers
best_tour = None
best_length = float('inf') # Use float for potentially large/inf values

# ----- Initial Best Tour Estimate (for tau_max) -----
# Simple Greedy NN from city 0
if num_cities > 0:
    initial_nn_tour = [0]
    visited_nn = {0}
    current_nn_city = 0
    initial_nn_length = 0.0 # Use float for calculation
    if num_cities > 1:
        while len(visited_nn) < num_cities:
            nearest_neighbor = -1
            min_dist = float('inf')
            for neighbor in range(num_cities):
                if neighbor not in visited_nn:
                    dist = dist_matrix[current_nn_city][neighbor]
                    if dist < min_dist:
                        min_dist = dist
                        nearest_neighbor = neighbor
            if nearest_neighbor != -1:
                initial_nn_tour.append(nearest_neighbor)
                visited_nn.add(nearest_neighbor)
                if min_dist != float('inf'): initial_nn_length += min_dist # Add distance if found
                current_nn_city = nearest_neighbor
            else: # Should not happen in complete graph
                 # Fallback: add remaining unvisited randomly
                 remaining = list(set(range(num_cities)) - visited_nn)
                 random.shuffle(remaining)
                 initial_nn_tour.extend(remaining)
                 initial_nn_length = calculate_tour_length(initial_nn_tour, dist_matrix) # Recalculate length
                 print("*** warning: Greedy NN failed, using random tour for init length.")
                 break
        # Close the tour
        if initial_nn_tour: # Check if tour is not empty
             initial_nn_length += dist_matrix[initial_nn_tour[-1]][initial_nn_tour[0]]
        else:
             initial_nn_length = 0

    # Apply 2-opt to the initial greedy tour for a better estimate
    initial_nn_tour, initial_nn_length = two_opt(initial_nn_tour, dist_matrix)
    initial_nn_length = float(initial_nn_length) # Ensure float

    if initial_nn_length <= 0: initial_nn_length = 1.0 # Avoid division by zero

    # ----- Initialize MMAS Pheromone Bounds -----
    tau_max = 1.0 / (rho * initial_nn_length)
    # Simple heuristic for tau_min
    tau_min = tau_max / (2.0 * num_cities) if num_cities > 0 else tau_max / 2.0
    tau_min = max(tau_min, 1e-9) # Ensure small positive floor
else:
    # Handle case of num_cities = 0
    initial_nn_length = 0
    tau_max = 1.0 # Assign default value
    tau_min = 0.5 # Assign default value

# Initialize pheromone matrix
pheromone = [[tau_max for _ in range(num_cities)] for _ in range(num_cities)]

last_update_iteration = 0

# ----- Main MMAS + 2-opt loop -----
for iteration in range(max_it):
    iteration_best_length = float('inf')
    iteration_best_tour = None

    # ---- construct tours ----
    for ant in range(num_ants):
        # Start ants randomly
        start_city = random.randrange(num_cities) if num_cities > 0 else 0
        if num_cities == 0: # Handle empty case
             tour_tmp = []
             visited_list = [] # Keep original variable name if needed
             current_city = 0
        else:
             visited_list = [False] * num_cities # Use list if sets cause issues with skeleton/validation
             visited_list[start_city] = True
             num_visited = 1
             tour_tmp = [start_city]
             current_city = start_city

        while num_visited < num_cities:
            unvisited_indices = [j for j, visited in enumerate(visited_list) if not visited]
            if not unvisited_indices: break # Safety break

            probabilities = []
            denom = 0.0
            for j in unvisited_indices:
                # Use clamped pheromone for selection probability calculation
                # current_pheromone = max(tau_min, min(tau_max, pheromone[current_city][j]))
                current_pheromone = pheromone[current_city][j] # Or use potentially unclamped

                tau = current_pheromone ** alpha
                eta = visibility[current_city][j] ** beta
                prob = tau * eta
                if prob < 0 or math.isnan(prob): prob = 0.0 # Handle invalid probabilities
                probabilities.append((j, prob))
                denom += prob

            if denom == 0.0 or math.isnan(denom) or math.isinf(denom):
                next_city = random.choice(unvisited_indices) # Fallback to random choice
            else:
                # Roulette wheel
                r = random.uniform(0, denom)
                cumulative = 0.0
                # Ensure probabilities list is not empty before accessing last element
                next_city = probabilities[-1][0] if probabilities else random.choice(unvisited_indices) 

                for city_idx, p_val in probabilities:
                    cumulative += p_val
                    if r <= cumulative:
                        next_city = city_idx
                        break
            
            tour_tmp.append(next_city)
            visited_list[next_city] = True
            num_visited += 1
            current_city = next_city

        # ---- Apply 2-opt ----
        if len(tour_tmp) == num_cities:
            optimized_tour, L = two_opt(tour_tmp, dist_matrix)
        else: # Handle incomplete tour
            optimized_tour = tour_tmp
            L = float('inf')


        # ---- Update Iteration/Global Bests ----
        if L < iteration_best_length:
            iteration_best_length = L
            iteration_best_tour = optimized_tour[:]

        if L < best_length:
            best_length = L
            best_tour = optimized_tour[:]
            stagnation_counter = 0
            last_update_iteration = iteration
            # Optional: print update
            # print(f"It {iteration+1}: New best {best_length}")
        else:
            # Increment stagnation only if a valid tour was found in this iteration
            if iteration_best_length != float('inf'):
                 stagnation_counter += 1


    # ---- Pheromone Evaporation & Lower Bound Clamp ----
    for i in range(num_cities):
        for j in range(num_cities):
            pheromone[i][j] *= (1 - rho)
            pheromone[i][j] = max(tau_min, pheromone[i][j]) # Apply lower bound

    # ---- MMAS Pheromone Deposit (Global Best) & Upper Bound Clamp ----
    # Use global best tour for update (a common MMAS variant)
    update_tour = best_tour
    update_length = best_length

    if update_tour is not None and update_length != float('inf') and update_length > 0:
        delta_tau = 1.0 / update_length
        for idx in range(num_cities):
            i = update_tour[idx]
            j = update_tour[(idx + 1) % num_cities] # Wrap around for last edge
            
            # Deposit and clamp with upper bound
            pheromone[i][j] += delta_tau
            pheromone[i][j] = min(tau_max, pheromone[i][j])
            pheromone[j][i] = pheromone[i][j] # Symmetric TSP


    # ---- Stagnation Handling ----
    if reset_pheromone_on_stagnation and stagnation_counter >= stagnation_limit:
        # print(f"--- Stagnation detected (Iter {iteration+1}). Resetting pheromones. ---")
        # Recalculate tau_max based on current best length for potentially better scaling
        if best_length > 0 and best_length != float('inf'):
             current_tau_max = 1.0 / (rho * best_length)
             current_tau_min = current_tau_max / (2.0 * num_cities) if num_cities > 0 else current_tau_max / 2.0
             current_tau_min = max(current_tau_min, 1e-9)
        else: # Fallback to initial values if best_length is invalid
            current_tau_max = tau_max
            current_tau_min = tau_min

        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] = current_tau_max # Reset to current tau_max
        stagnation_counter = 0 # Reset counter
        # Update global tau_max and tau_min if recalculated
        tau_max = current_tau_max
        tau_min = current_tau_min


# Final assignment to reserved variables
if best_tour is None and num_cities > 0:
     # If no tour ever found, create a default one (e.g., 0, 1, 2...)
     print("*** Warning: No valid tour found. Creating default tour.")
     tour = list(range(num_cities))
     tour_length = calculate_tour_length(tour, dist_matrix)
elif best_tour is None and num_cities == 0:
     tour = []
     tour_length = 0
else:
     tour = best_tour
     tour_length = int(round(best_length)) # Ensure integer


# Update added_note (ensure this variable is defined before this block in the skeleton)
added_note = "Implementation of Max-Min Ant System (MMAS) with 2-opt local search enhancement." # Reset note first
added_note += f" Params: alpha={alpha:.2f}, beta={beta:.2f}, rho={rho:.2f}."
# Report the *final* tau values used, reflecting potential resets
added_note += f" Final tau_max≈{tau_max:.4g}, tau_min≈{tau_min:.4g}."
added_note += f" Stagnation reset: {reset_pheromone_on_stagnation} (limit={stagnation_limit})."

# ----- End of MMAS + 2-opt Implementation -----









############ START OF SECTOR 10 (IGNORE THIS COMMENT)
############
############ YOUR CODE SHOULD NOW BE COMPLETE AND WHEN EXECUTION OF THIS PROGRAM 'skeleton.py'
############ REACHES THIS POINT, YOU SHOULD HAVE COMPUTED A TOUR IN THE RESERVED LIST VARIABLE 'tour', 
############ WHICH HOLDS A LIST OF THE INTEGERS FROM {0, 1, ..., 'num_cities' - 1} SO THAT EVERY INTEGER
############ APPEARS EXACTLY ONCE, AND YOU SHOULD ALSO HOLD THE LENGTH OF THIS TOUR IN THE RESERVED
############ INTEGER VARIABLE 'tour_length'.
############
############ YOUR TOUR WILL BE PACKAGED IN A TOUR FILE OF THE APPROPRIATE FORMAT AND THIS TOUR FILE'S
############ NAME WILL BE A MIX OF THE NAME OF THE CITY FILE, THE NAME OF THIS PROGRAM AND THE
############ CURRENT DATE AND TIME. SO, EVERY SUCCESSFUL EXECUTION GIVES A TOUR FILE WITH A UNIQUE
############ NAME AND YOU CAN RENAME THE ONES YOU WANT TO KEEP LATER.
############
############ DO NOT EDIT ANY TOUR FILE! ALL TOUR FILES MUST BE LEFT AS THEY WERE ON OUTPUT.
############
############ DO NOT TOUCH OR ALTER THE CODE BELOW THIS POINT! YOU HAVE BEEN WARNED!
############

end_time = time.time()
elapsed_time = round(end_time - start_time, 1)

if algorithm_code == "GA":
    try: max_it
    except NameError: max_it = None
    try: pop_size
    except NameError: pop_size = None
    if added_note != "":
        added_note = added_note + "\n"
    added_note = added_note + "The parameter values are 'max_it' = " + str(max_it) + " and 'pop_size' = " + str(pop_size) + "."

if algorithm_code == "AC":
    try: max_it
    except NameError: max_it = None
    try: num_ants
    except NameError: num_ants = None
    if added_note != "":
        added_note = added_note + "\n"
    added_note = added_note + "The parameter values are 'max_it' = " + str(max_it) + " and 'num_ants' = " + str(num_ants) + "."

if algorithm_code == "PS":
    try: max_it
    except NameError: max_it = None
    try: num_parts
    except NameError: num_parts = None
    if added_note != "":
        added_note = added_note + "\n"
    added_note = added_note + "The parameter values are 'max_it' = " + str(max_it) + " and 'num_parts' = " + str(num_parts) + "."
    
added_note = added_note + "\nRUN-TIME = " + str(elapsed_time) + " seconds.\n"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
added_note = added_note + "DATE-TIME = " + dt_string + ".\n"

flag = "good"
length = len(tour)
for i in range(0, length):
    if isinstance(tour[i], int) == False:
        flag = "bad"
    else:
        tour[i] = int(tour[i])
if flag == "bad":
    print("*** error: Your tour contains non-integer values.")
    sys.exit()
if isinstance(tour_length, int) == False:
    print("*** error: The tour-length is a non-integer value.")
    sys.exit()
tour_length = int(tour_length)
if len(tour) != num_cities:
    print("*** error: The tour does not consist of " + str(num_cities) + " cities as there are, in fact, " + str(len(tour)) + ".")
    sys.exit()
flag = "good"
for i in range(0, num_cities):
    if not i in tour:
        flag = "bad"
if flag == "bad":
    print("*** error: Your tour has illegal or repeated city names.")
    sys.exit()
check_tour_length = 0
for i in range(0, num_cities - 1):
    check_tour_length = check_tour_length + dist_matrix[tour[i]][tour[i + 1]]
check_tour_length = check_tour_length + dist_matrix[tour[num_cities - 1]][tour[0]]
if tour_length != check_tour_length:
    print("*** error: The length of your tour is not " + str(tour_length) + "; it is actually " + str(check_tour_length) + ".")
    sys.exit()
print("You, user " + my_user_name + ", have successfully built a tour of length " + str(tour_length) + "!")
len_user_name = len(my_user_name)
user_number = 0
for i in range(0, len_user_name):
    user_number = user_number + ord(my_user_name[i])
alg_number = ord(algorithm_code[0]) + ord(algorithm_code[1])
len_dt_string = len(dt_string)
date_time_number = 0
for i in range(0, len_dt_string):
    date_time_number = date_time_number + ord(dt_string[i])
tour_diff = abs(tour[0] - tour[num_cities - 1])
for i in range(0, num_cities - 1):
    tour_diff = tour_diff + abs(tour[i + 1] - tour[i])
certificate = user_number + alg_number + date_time_number + tour_diff
local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]
output_file_name = script_name[0:len(script_name) - 3] + "_" + input_file[0:len(input_file) - 4] + "_" + output_file_time + ".txt"

f = open(output_file_name,'w')
f.write("USER = {0} ({1} {2}),\n".format(my_user_name, my_first_name, my_last_name))
f.write("ALGORITHM CODE = {0}, NAME OF CITY-FILE = {1},\n".format(algorithm_code, input_file))
f.write("SIZE = {0}, TOUR LENGTH = {1},\n".format(num_cities, tour_length))
f.write(str(tour[0]))
for i in range(1,num_cities):
    f.write(",{0}".format(tour[i]))
f.write(",\nNOTE = {0}".format(added_note))
f.write("CERTIFICATE = {0}.\n".format(certificate))
f.close()
print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")

############ END OF SECTOR 10 (IGNORE THIS COMMENT)
