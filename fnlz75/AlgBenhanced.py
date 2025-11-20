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

input_file = "AISearchfile048.txt"

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

algorithm_code = "PS"

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



import random, math, time
from typing import List, Tuple

# ------------------------- helper utilities -------------------------------
City = int
Tour = List[City]
Matrix = List[List[float]]
EPS = 1e-9

def length_of(tour: Tour, dist: Matrix) -> float:
    return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

# ------------------------- 3‑Opt (deterministic) --------------------------

def build_candidate_lists(dist: Matrix, candidate_size: int = 20) -> List[List[int]]:
    
    #For each city i, build a list of its candidate_size nearest neighbors.
    
    n = len(dist)
    candidates = []
    for i in range(n):
        # Create list of other cities sorted by distance
        nbrs = [(j, dist[i][j]) for j in range(n) if j != i]
        nbrs.sort(key=lambda x: x[1])  # Sort by distance
        candidates.append([j for j, _ in nbrs[:candidate_size]])
    return candidates

def delta_3opt(tour: Tour, dist: Matrix, i: int, j: int, k: int) -> Tuple[float, int]:
    """
    Compute the best 3-opt delta (improvement) for indices i<j<k.
    Returns (best_gain, move_type) where move_type is 0-7, or (0, -1) if no improvement.
    """
    n = len(tour)
    # Get cities at each position and their neighbors in the tour
    a, b = tour[i], tour[(i+1) % n]
    c, d = tour[j], tour[(j+1) % n]
    e, f = tour[(k) % n], tour[(k+1) % n]
    
    # Original edges that will be potentially removed
    d_ab = dist[a][b]
    d_cd = dist[c][d]
    d_ef = dist[e][f]
    
    # Cost of removing three edges
    removed = d_ab + d_cd + d_ef
    
    # Calculate costs for all possible reconnections
    # Option 0: Original tour (no change)
    # Option 1: Reverse segment i+1 to j
    gain1 = removed - (dist[a][c] + dist[b][d] + dist[e][f])
    
    # Option 2: Reverse segment j+1 to k
    gain2 = removed - (dist[a][b] + dist[c][e] + dist[d][f])
    
    # Option 3: Reverse both segments
    gain3 = removed - (dist[a][c] + dist[b][e] + dist[d][f])
    
    # Option 4: Replace (a,b) and (c,d) with (a,d) and (b,c)
    gain4 = removed - (dist[a][d] + dist[b][c] + dist[e][f])
    
    # Options 5-7: Other possible reconnections
    gain5 = removed - (dist[a][e] + dist[b][d] + dist[c][f])
    gain6 = removed - (dist[a][d] + dist[b][f] + dist[c][e])
    gain7 = removed - (dist[a][e] + dist[b][c] + dist[d][f])
    
    # Find the best option
    gains = [0, gain1, gain2, gain3, gain4, gain5, gain6, gain7]
    best_gain = max(gains)
    best_move = gains.index(best_gain)
    
    # Only return a move if it improves the tour
    if best_gain <= EPS:
        return 0, -1
    return best_gain, best_move

def apply_3opt_move(tour: Tour, i: int, j: int, k: int, move_type: int) -> Tour:
    
    #Apply a 3-opt move of the specified type to the tour.
    
    n = len(tour)
    # Extract segments
    a, b = i, (i+1) % n
    c, d = j, (j+1) % n
    e, f = k, (k+1) % n
    
    # Normalize indices to handle wrap-around
    # Ensure i < j < k < n for segment extraction
    normalized_indices = []
    for idx in [i, j, k]:
        if idx >= n:
            normalized_indices.append(idx - n)
        else:
            normalized_indices.append(idx)
    
    i, j, k = sorted(normalized_indices)
    
    # Extract segments
    seg1 = tour[:i+1]
    seg2 = tour[i+1:j+1]
    seg3 = tour[j+1:k+1]
    seg4 = tour[k+1:] if k < n-1 else []
    
    # Apply the move based on move_type
    if move_type == 1:   # Reverse seg2
        return seg1 + seg2[::-1] + seg3 + seg4
    elif move_type == 2: # Reverse seg3
        return seg1 + seg2 + seg3[::-1] + seg4
    elif move_type == 3: # Reverse both seg2 and seg3
        return seg1 + seg2[::-1] + seg3[::-1] + seg4
    elif move_type == 4: # Swap seg2 and seg3
        return seg1 + seg3 + seg2 + seg4
    elif move_type == 5: # Swap and reverse combinations
        return seg1 + seg3[::-1] + seg2 + seg4
    elif move_type == 6:
        return seg1 + seg3 + seg2[::-1] + seg4
    elif move_type == 7:
        return seg1 + seg3[::-1] + seg2[::-1] + seg4
    else:
        return tour  # No change

def three_opt(tour: Tour, dist: Matrix, max_time_seconds: float = 300.0) -> Tuple[Tour, float]:
    
    #Enhanced 3-opt algorithm with candidate lists and delta evaluation

    n = len(tour)
    start_time = time.time()
    
    # Adjust candidate list size based on problem size
    candidate_size = min(30, max(15, n // 10))
    candidates = build_candidate_lists(dist, candidate_size)
    
    # Initialize best tour
    best = tour[:]
    best_len = length_of(best, dist)
    
    # Loop until no more improvements or time limit
    iteration = 0
    total_improvements = 0
    
    while True:
        iteration += 1
        improved = False
        improvements_this_iter = 0
        
        # Check time limit
        if time.time() - start_time > max_time_seconds:
            print(f"Time limit reached after {iteration} iterations")
            break
        
        # For each city, try 3-opt moves using candidate lists
        for i in range(n):
            # Look at the neighbors of the current city in the tour
            current = best[i]
            next_city = best[(i+1) % n]
            
            # Try to replace the edge (current, next_city) using candidate lists
            for c1 in candidates[current]:
                # Skip if c1 is already the next city
                if c1 == next_city:
                    continue
                
                # Find j where best[j] == c1
                try:
                    j = best.index(c1)
                except ValueError:
                    continue
                
                # Ensure i < j
                if j <= i:
                    continue
                
                # Now try to find a third city from the candidates
                for c2 in candidates[best[j]]:
                    # Find k where best[k] == c2
                    try:
                        k = best.index(c2)
                    except ValueError:
                        continue
                    
                    # Ensure j < k
                    if k <= j or k <= i:
                        continue
                    
                    # Calculate delta and move type
                    gain, move_type = delta_3opt(best, dist, i, j, k)
                    
                    # If improvement found, apply it
                    if gain > EPS and move_type >= 0:
                        new_tour = apply_3opt_move(best, i, j, k, move_type)
                        new_len = length_of(new_tour, dist)
                        
                        # Verify the improvement
                        if new_len < best_len - EPS:
                            best = new_tour
                            best_len = new_len
                            improved = True
                            improvements_this_iter += 1
                            total_improvements += 1
                            # Break inner loop to restart with the new tour
                            break
                
                # Break middle loop if improved
                if improved:
                    break
            
            # Break outer loop if improved
            if improved:
                break
        
        # Report progress every iteration
        elapsed = time.time() - start_time
        
        # Stop if no improvements in this iteration
        if not improved:
            break
        
    
    # Report final results
    elapsed = time.time() - start_time
    print(f"3-opt completed with {total_improvements} total improvements in {elapsed:.2f} seconds")
    return best, best_len

# ------------------------- Ant Colony (classic) ---------------------------

def nearest_neighbor(dist: Matrix) -> float:
    n = len(dist)
    best = math.inf
    for s in range(n):
        unv = set(range(n)) - {s}
        cur, l = s, 0.0
        while unv:
            # Use a small value instead of zero for distance
            nxt = min(unv, key=lambda j: max(0.1, dist[cur][j]))
            l += max(0.1, dist[cur][nxt])  # Avoid adding zeros
            unv.remove(nxt)
            cur = nxt
        best = min(best, l)
    return best

def ant_colony(alpha: float, beta: float, *, ants: int = 10, iterations: int = 200, rho: float = 0.1, Q: float = 1.0) -> Tuple[Tour, float]:
    n = num_cities

    
    # Precompute heuristic information (inverse distance)
    eta = [[0 if i == j else (1.0 / max(0.1, dist_matrix[i][j])) for j in range(n)] for i in range(n)]
    
    # Initialize pheromone trails
    tau0 = 1.0 / (n * max(0.1, nearest_neighbor(dist_matrix)))  # Avoid div by zero
    tau = [[tau0] * n for _ in range(n)]
    
    # Track best tour
    best_tour, best_len = None, math.inf
    
    
    # Build candidate lists for efficient tour construction
    candidate_size = min(30, max(20, n // 5))
    candidates = build_candidate_lists(dist_matrix, candidate_size)
    
    
    for iteration in range(iterations):
        
            
            
        tours, lengths = [], []
        for ant in range(ants):
            start = random.randrange(n)
            tour = [start]
            unv = set(range(n)) - {start}
            cur = start
            
            while unv:
                # First try using candidate list (90% of the time)
                if random.random() < 0.9:
                    # Filter candidates that are still unvisited
                    available_candidates = [c for c in candidates[cur] if c in unv]
                    
                    if available_candidates:
                        probs, total = [], 0.0
                        
                        for j in available_candidates:
                            # Make sure we don't get NaN from 0^0 when alpha or beta is 0
                            tau_val = max(1e-10, tau[cur][j]) ** alpha if alpha > 0 else 1.0
                            eta_val = max(1e-10, eta[cur][j]) ** beta if beta > 0 else 1.0
                            p = tau_val * eta_val
                            probs.append((j, p))
                            total += p
                        
                        # Select next city using candidates
                        if total > 1e-10:
                            r = random.random() * total
                            cum = 0.0
                            nxt = available_candidates[0]  # Default
                            for city, p in probs:
                                cum += p
                                if cum >= r:
                                    nxt = city
                                    break
                            
                            tour.append(nxt)
                            unv.remove(nxt)
                            cur = nxt
                            continue
                
                # Fallback: check all remaining unvisited cities
                cities_to_check = list(unv)
                
                    
                probs, total = [], 0.0
                for j in cities_to_check:
                    tau_val = max(1e-10, tau[cur][j]) ** alpha if alpha > 0 else 1.0
                    eta_val = max(1e-10, eta[cur][j]) ** beta if beta > 0 else 1.0
                    p = tau_val * eta_val
                    probs.append((j, p))
                    total += p
                
                # If total is too small, just pick randomly
                if total < 1e-10:
                    nxt = random.choice(list(unv))
                else:
                    r = random.random() * total
                    cum = 0.0
                    nxt = cities_to_check[0]  # Default
                    for city, p in probs:
                        cum += p
                        if cum >= r:
                            nxt = city
                            break
                tour.append(nxt)
                unv.remove(nxt)
                cur = nxt
            
            # Calculate tour length
            l = length_of(tour, dist_matrix)
            tours.append(tour)
            lengths.append(l)
            
            if l < best_len:
                best_tour, best_len = tour, l
                
            
                    
        
        # Pheromone update with optimizations for large instances
        # Evaporation phase
        evap = 1 - rho
        for i in range(n):
            for j in range(i, n):  # Use symmetry to reduce computations
                tau[i][j] *= evap
                tau[j][i] = tau[i][j]  # Mirror updates
        
        # Deposit phase - use all tours regardless of instance size
        for tour, l in zip(tours, lengths):
            delta = Q / l
            for i in range(n):
                a, b = tour[i], tour[(i + 1) % n]
                tau[a][b] += delta
                tau[b][a] += delta  # Symmetric update
        
        # Continue running regardless of instance size

    
    
    return best_tour, best_len

# ---------------------- Particle Swarm (2‑D search) -----------------------

def pso_find_alpha_beta(*, swarm: int = 10, iters: int = 30, w: float = 0.7, c1: float = 2.0, c2: float = 2.0):
    """PSO for finding ideal alpha/beta ACO parameters"""
    
    n = num_cities
    
    # Bounds as in the paper: α, β ∈ [0,2]
    ALB, AUB = 0.0, 2.0
    BLB, BUB = 0.0, 2.0
    
    particles = []
   
    
    # Initialize particles with random positions
    for i in range(swarm):
        a = random.uniform(ALB, AUB)
        b = random.uniform(BLB, BUB)
        
            
        va, vb = random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)
        particles.append({'pos':[a,b], 'vel':[va,vb], 'best':[a,b], 'best_val':math.inf})
        
    gbest, gbest_val = [None, None], math.inf
    
    
    # Use consistent ACO iterations 
    aco_iterations = 40
    
    for iteration in range(iters):
        
        
        # Evaluate all particles 
        particles_to_evaluate = particles
            
        for p_idx, p in enumerate(particles_to_evaluate):
            # Evaluate all particles
                
           
            aco_start = time.time()
            _, val = ant_colony(p['pos'][0], p['pos'][1], ants=10, iterations=aco_iterations)
            aco_time = time.time() - aco_start
            
      
            
            if val < p['best_val']:
                p['best_val'], p['best'] = val, p['pos'][:]
            
                
                if val < gbest_val:
                    gbest_val, gbest = val, p['pos'][:]
                
                    
                
        # Update particle positions with adaptive velocity control
        # Reduce velocity magnitude as iterations progress for finer search
        velocity_damper = max(0.5, 1.0 - (iteration / iters))
        
        for p in particles:
            for d in (0,1):
                r1, r2 = random.random(), random.random()
                p['vel'][d] = (w * p['vel'][d] + c1 * r1 * (p['best'][d] - p['pos'][d]) +
                              c2 * r2 * (gbest[d] - p['pos'][d]))
                # Apply damping factor to velocity
                p['vel'][d] *= velocity_damper
                p['pos'][d] += p['vel'][d]
            p['pos'][0] = max(ALB, min(AUB, p['pos'][0]))
            p['pos'][1] = max(BLB, min(BUB, p['pos'][1]))
            
        
        
    
    
 

        
    
    return gbest[0], gbest[1]

# --------------------------- MAIN BODY ------------------------------------

seed = int(time.time()) % 2**32
random.seed(seed)

# Initialize algorithm parameters
max_it = 800   # ACO iterations after PSO phase
num_parts = 10  # PSO swarm size
num_ants = 10   # Ants per ACO iteration




# 2. Stage 1 – PSO searches best (alpha, beta)
start_pso = time.time()

# Run PSO to find best parameters regardless of instance size
alpha, beta = pso_find_alpha_beta(swarm=num_parts, iters=20)
pso_time = time.time() - start_pso


# 3. Stage 2 – Run a longer ACO with that pair
start_aco = time.time()
best_tour, best_len = ant_colony(alpha, beta, ants=num_ants, iterations=max_it)
aco_time = time.time() - start_aco


# 4. Stage 3 – Apply optimized 3‑Opt
start_3opt = time.time()

# Set a consistent time limit for 3-opt regardless of problem size
opt_time_limit = 300  # Use full 5 minutes for all instances


best_tour, best_len = three_opt(best_tour, dist_matrix, max_time_seconds=opt_time_limit)
opt_time = time.time() - start_3opt


# 5. Deliver reserved variables
tour = best_tour
tour_length = int(round(best_len))

# Print total execution breakdown
total_time = pso_time + aco_time + opt_time
print(f"\n[EXECUTION SUMMARY]")
print(f"PSO phase: {pso_time:.2f}s ({(pso_time/total_time)*100:.1f}% of total)")
print(f"ACO phase: {aco_time:.2f}s ({(aco_time/total_time)*100:.1f}% of total)")
print(f"3-opt phase: {opt_time:.2f}s ({(opt_time/total_time)*100:.1f}% of total)")
print(f"Total algorithm time: {total_time:.2f}s")

# Optional: annotate
added_note += (f"\nEnhanced PSO‑ACO‑3Opt: α={alpha:.3f}, β={beta:.3f}, "
               f"seed={seed}, max_it={max_it}, ants={num_ants}, swarm={num_parts}. "
               f"Uses candidate lists and delta evaluation with consistent behavior for all city sizes.")





















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