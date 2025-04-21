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

from __future__ import annotations
import os
import sys
import time
import random
from datetime import datetime
import math 
from typing import List, Tuple, Dict

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

input_file = "AISearchfile058.txt"

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


# --- Type Definitions ---
Tour      = List[int]
Swap      = Tuple[int, int]      # arbitrary transposition (city_i, city_j)
Velocity  = List[Swap]           # Sequence of swaps defining movement
Particles = List["Particle"]     # List of particles
Pheromones = List[List[float]]   # Pheromone matrix

# --- ACO Parameters (Add these) ---
pheromone_rho = 0.1  # Pheromone evaporation rate (0 < rho <= 1)
pheromone_q = 100.0  # Pheromone deposit constant (adjust based on typical tour lengths)
pheromone_alpha = 1.0 # Influence of pheromone in biased target generation
pheromone_beta = 2.0  # Influence of heuristic (distance) in biased target generation
pheromone_initial = 0.1 # Initial pheromone level

# --- PSO Parameters (Adjust as needed) ---
max_it = 1000
num_parts = num_cities * 5  # Can adjust based on tuning
theta = 0.75   # Inertia weight (tuned slightly)
alpha = 1.5    # Cognitive parameter
beta = 1.5     # Social parameter
delta = 2      # Ring topology (changed from infinity)

# ---------------------------- Helper Functions (Mostly from Basic) ----------------------------- #

def canonical(tour: Tour) -> Tour:
    """Rotate so city 0 is first; return *copy*."""
    k = tour.index(0)
    return tour[k:] + tour[:k]

def tour_length(tour: Tour, d) -> int:
    """Calculate the length of a tour."""
    length = 0
    num = len(tour)
    for i in range(num):
        u, v = tour[i], tour[(i + 1) % num]
        length += d[u][v]
    return length

# ---- Velocity definition using arbitrary swaps (from Basic) ---- #

def permutation_from_tours(a: Tour, b: Tour) -> Dict[int, int]:
    return {a[i]: b[i] for i in range(len(a))}

def cycle_decomposition(perm: Dict[int, int]) -> List[List[int]]:
    cycles = []
    seen = set()
    for x in perm:
        if x in seen or perm[x] == x: continue
        cycle = [x]
        y = perm[x]
        while y != x:
            cycle.append(y)
            y = perm[y]
        seen.update(cycle)
        cycles.append(cycle)
    cycles.sort(key=lambda c: (-len(c), min(c)))
    return cycles

def velocity_from_cycles(cycles: List[List[int]]) -> Velocity:
    v: Velocity = []
    for cyc in cycles:
        for i in range(len(cyc) - 1): v.append((cyc[i], cyc[i + 1]))
    return v

def arbitrary_velocity(a: Tour, b: Tour) -> Velocity:
    """Calculates velocity as swaps based on cycle decomposition."""
    perm = permutation_from_tours(a, b)
    cycles = cycle_decomposition(perm)
    return velocity_from_cycles(cycles)

# ---- Velocity algebra (from Basic) ---- #

def apply_velocity(tour: Tour, v: Velocity) -> Tour:
    """Applies a velocity (sequence of swaps) to a tour."""
    tour = tour[:]
    pos = {city: idx for idx, city in enumerate(tour)}
    for c1, c2 in v:
        if c1 in pos and c2 in pos: # Check if cities are still in expected positions
             i, j = pos[c1], pos[c2]
             tour[i], tour[j] = tour[j], tour[i]
             pos[c1], pos[c2] = j, i
    return tour

def velocity_add(v1: Velocity, v2: Velocity) -> Velocity:
    return v1 + v2

def scalar_multiply(v: Velocity, gamma: float) -> Velocity:
    """Scales a velocity."""
    if gamma <= 0 or not v: return []
    if gamma < 1:
        k = int(math.floor(gamma * len(v)))
        return v[:k]
    k = int(math.floor(gamma))
    frac = gamma - k
    return v * k + scalar_multiply(v, frac)

# --- 2-opt Local Search (NEW Enhancement) ---
def two_opt_local_search(tour: Tour, d, max_iterations: int = 100) -> Tuple[Tour, int]:
    """Apply 2-opt local search to improve a tour."""
    n = len(tour)
    best_tour = tour[:]
    best_len = tour_length(tour, d)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, n-2):  # Skip 0 to keep tour canonical
            for j in range(i+1, n):
                if j-i == 1:  # Skip adjacent cities, no benefit
                    continue
                    
                # Consider reversing segment from i to j
                # In 2-opt, we break edges (i-1,i) and (j,j+1) and reconnect with (i-1,j) and (i,j+1)
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_len = tour_length(new_tour, d)
                
                if new_len < best_len:
                    best_tour = new_tour
                    best_len = new_len
                    improved = True
                    break  # First improvement strategy - restart when improvement found
            
            if improved:
                break
    
    return best_tour, best_len

# --- Simulated Annealing Acceptance Criterion (NEW Enhancement) ---
def accept_with_sa(current_len: int, new_len: int, temperature: float) -> bool:
    """Decide whether to accept a new solution using simulated annealing criterion."""
    if new_len <= current_len:
        return True  # Always accept better solutions
    
    # Probabilistically accept worse solutions based on temperature and delta
    delta = new_len - current_len
    probability = math.exp(-delta / temperature)
    return random.random() < probability

# --- Pheromone-Guided Target Generation (NEW Enhancement) ---

def generate_pheromone_biased_target(
    base_tour: Tour,
    pheromones: Pheromones,
    d,
    num_swaps: int = 3 # Number of biased swaps to attempt
) -> Tour:
    """Creates a slightly modified target tour biased by pheromones."""
    target_tour = base_tour[:]
    n = len(target_tour)
    pos = {city: idx for idx, city in enumerate(target_tour)}

    for _ in range(num_swaps):
        # Select two distinct edges (i, i+1) and (j, j+1) to potentially swap (2-opt style)
        # Avoid index 0 to keep tour canonical if possible, indices are relative to current target_tour
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)
        if j == n - 1 and i == 1: continue # Avoid reversing the whole tour except city 0

        city_i, city_i1 = target_tour[i], target_tour[(i + 1) % n]
        city_j, city_j1 = target_tour[j], target_tour[(j + 1) % n]

        # Calculate desirability of current edges vs swapped edges
        # Higher value means more desirable
        current_desirability = (pheromones[city_i][city_i1] ** pheromone_alpha) * ((1.0 / max(1, d[city_i][city_i1])) ** pheromone_beta) + \
                               (pheromones[city_j][city_j1] ** pheromone_alpha) * ((1.0 / max(1, d[city_j][city_j1])) ** pheromone_beta)

        # Desirability after swapping edges (i, j+1) and (j, i+1)
        # Note: This corresponds to a 2-opt move reversing the segment between i+1 and j
        swapped_desirability = (pheromones[city_i][city_j] ** pheromone_alpha) * ((1.0 / max(1, d[city_i][city_j])) ** pheromone_beta) + \
                               (pheromones[city_i1][city_j1] ** pheromone_alpha) * ((1.0 / max(1, d[city_i1][city_j1])) ** pheromone_beta)

        # Probabilistically decide whether to perform the 2-opt swap
        # Favor swaps that lead to higher pheromone/shorter distance edges
        probability_to_swap = swapped_desirability / (current_desirability + swapped_desirability + 1e-9) # Add epsilon for stability

        if random.random() < probability_to_swap:
            # Perform the 2-opt swap (reverse segment between i+1 and j)
            segment_to_reverse = target_tour[i+1 : j+1]
            segment_to_reverse.reverse()
            target_tour = target_tour[:i+1] + segment_to_reverse + target_tour[j+1:]
            # Update position map needed if more swaps are done
            pos = {city: idx for idx, city in enumerate(target_tour)}

    return target_tour


# ---------------------------- Particle class (Unchanged from Basic) ------------------------------- #
class Particle:
    def __init__(self, init_tour: Tour, d):
        self.d = d
        self.tour: Tour = canonical(init_tour)
        self.velocity: Velocity = []
        self.best_tour: Tour = self.tour[:]
        self.best_len  = tour_length(self.tour, d)
        # Stagnation tracking
        self.stagnation_count = 0
        self.last_improvement = 0
        self.stagnation_threshold = 50  # Number of iterations without improvement before considered stagnant

    def move(self):
        """Applies velocity and updates personal best."""
        self.tour = apply_velocity(self.tour, self.velocity)
        length = tour_length(self.tour, self.d)
        if length < self.best_len:
            self.best_len = length
            self.best_tour = self.tour[:]
            self.stagnation_count = 0  # Reset stagnation counter
            return True  # Improved
        else:
            self.stagnation_count += 1  # Increment stagnation counter
            return False  # No improvement

    def is_stagnant(self, current_iteration: int) -> bool:
        """Check if particle is stagnant."""
        return self.stagnation_count >= self.stagnation_threshold
        
    def reset(self, n: int, best_tour: Tour = None):
        """Reset particle with random tour or perturbed version of best tour."""
        if best_tour and random.random() < 0.5:  # 50% chance to use best_tour as base
            # Create a perturbed version of the best tour
            self.tour = best_tour[:]
            # Apply several random swaps
            for _ in range(int(n * 0.3)):  # Perturb about 30% of the cities
                i, j = random.sample(range(n), 2)
                self.tour[i], self.tour[j] = self.tour[j], self.tour[i]
            self.tour = canonical(self.tour)
        else:
            # Create a completely new random tour
            t = list(range(n))
            random.shuffle(t)
            self.tour = canonical(t)
            
        # Reset velocity with random swaps
        m = random.randint(1, n)
        self.velocity = []
        for _ in range(m):
            c1_idx, c2_idx = random.sample(range(1, n), 2)
            self.velocity.append((self.tour[c1_idx], self.tour[c2_idx]))
            
        # Reset counters but keep personal best
        self.stagnation_count = 0

# ---------------------------- Main ACO-PSO Hybrid Routine ----------------------------- #

def aco_pso_tsp(
    d,
    *,
    num_particles: int = 150,
    max_iter: int = 1000,
    theta: float = 0.8,
    alpha: float = 1.5,
    beta: float = 1.5,
    delta: int | float = float("inf"),
    rho: float = 0.1,  # Pheromone evaporation rate
    q: float = 100.0,  # Pheromone deposit constant
    initial_pheromone: float = 0.1, # Initial pheromone level
    local_search_freq: int = 20,  # Apply local search every N iterations
    time_limit: float | None = None,
    seed: int | None = None,
) -> Tuple[Tour, int]:
    """Enhanced Hybrid ACO-PSO for TSP with anti-stagnation strategies. Returns (best_tour, length)."""
    if seed is not None:
        random.seed(seed)

    n = len(d)

    # --- Define multiple swarm parameter sets for diversification ---
    swarm_configs = [
        {"theta": 0.9, "alpha": 1.8, "beta": 1.2},  # Exploration-focused
        {"theta": 0.7, "alpha": 1.5, "beta": 1.5},  # Balanced
        {"theta": 0.5, "alpha": 1.2, "beta": 1.8},  # Exploitation-focused
    ]
    
    swarm_size = num_particles // len(swarm_configs)
    remaining = num_particles % len(swarm_configs)
    swarm_sizes = [swarm_size + (1 if i < remaining else 0) for i in range(len(swarm_configs))]
    
    # --- Initialize Pheromone Matrix ---
    pheromones: Pheromones = [[initial_pheromone for _ in range(n)] for _ in range(n)]

    # --- Initialize PSO Swarms with Different Parameters ---
    all_particles: Particles = []
    swarm_particles: List[Particles] = []
    
    start_idx = 0
    for swarm_idx, (size, config) in enumerate(zip(swarm_sizes, swarm_configs)):
        swarm = []
        for _ in range(size):
            t = list(range(n))
            random.shuffle(t)
            t = canonical(t)
            p = Particle(t, d)
            m = random.randint(1, n)
            vel: Velocity = []
            for _ in range(m):
                c1_idx, c2_idx = random.sample(range(1, n), 2)
                vel.append((t[c1_idx], t[c2_idx]))
            p.velocity = vel
            swarm.append(p)
            all_particles.append(p)
        swarm_particles.append(swarm)
    
    # --- Initialize Global Best ---
    gbest_tour, gbest_len = min(((p.best_tour, p.best_len) for p in all_particles), key=lambda x: x[1])
    
    # --- Track best tours per swarm ---
    swarm_best_tours = []
    swarm_best_lens = []
    for swarm in swarm_particles:
        best_tour, best_len = min(((p.best_tour, p.best_len) for p in swarm), key=lambda x: x[1])
        swarm_best_tours.append(best_tour)
        swarm_best_lens.append(best_len)
    
    # --- Track improvement history for adaptive parameters ---
    stagnation_count = 0
    last_improvement_iter = 0
    
    # --- Initialize temperature for simulated annealing ---
    initial_temp = 100.0
    final_temp = 0.1
    cooling_rate = -math.log(final_temp / initial_temp) / max_iter
    
    # --- Main Loop ---
    start = time.time()
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start) >= time_limit:
            break
        
        # --- Update temperature for simulated annealing ---
        temperature = initial_temp * math.exp(-cooling_rate * it)
        
        # --- Determine if global stagnation is occurring ---
        global_stagnant = it - last_improvement_iter > 100  # No improvement for 100 iterations
        
        # --- Process each swarm with its specific parameters ---
        for swarm_idx, (swarm, config) in enumerate(zip(swarm_particles, swarm_configs)):
            # Adapt parameters based on stagnation
            current_theta = config["theta"]
            current_alpha = config["alpha"]
            current_beta = config["beta"]
            
            # Increase exploration when stagnant, increase exploitation otherwise
            if global_stagnant:
                current_theta = min(0.95, config["theta"] + 0.05)  # Increase inertia
                current_alpha = max(1.0, config["alpha"] - 0.1)    # Decrease cognitive
                current_beta = max(1.0, config["beta"] - 0.1)      # Decrease social
            
            # Track if any particle in this swarm improved
            swarm_improved = False
            
            # --- Update Every Particle in this Swarm ---
            for idx, p in enumerate(swarm):
                # Reset stagnant particles
                if p.is_stagnant(it):
                    # Use different strategies based on swarm index
                    if swarm_idx == 0:  # Exploration swarm - complete reset
                        p.reset(n)
                    elif swarm_idx == len(swarm_configs) - 1:  # Exploitation swarm - reset based on best
                        p.reset(n, gbest_tour)
                    else:  # Middle swarms - mix of strategies
                        # Reset based on other swarm bests occasionally for information exchange
                        other_swarm_idx = random.randint(0, len(swarm_configs) - 1)
                        while other_swarm_idx == swarm_idx and len(swarm_configs) > 1:
                            other_swarm_idx = random.randint(0, len(swarm_configs) - 1)
                        p.reset(n, swarm_best_tours[other_swarm_idx])
                
                # Determine Neighborhood Best for this particle
                if math.isinf(delta):
                    nbest_tour_base = swarm_best_tours[swarm_idx]  # Base target is swarm best
                else:
                    d_int = int(delta)
                    ring = [(idx + i) % len(swarm) for i in range(-d_int, d_int + 1)]
                    nbest_tour_base, _ = min(((swarm[j].best_tour, swarm[j].best_len) for j in ring), key=lambda x: x[1])
                
                # Generate Pheromone-Biased Targets
                pbest_target = generate_pheromone_biased_target(p.best_tour, pheromones, d)
                nbest_target = generate_pheromone_biased_target(nbest_tour_base, pheromones, d)
                
                # Standard PSO Velocity Update with adaptive parameters
                v_inertia = scalar_multiply(p.velocity, current_theta)
                v_cognitive = scalar_multiply(arbitrary_velocity(p.tour, pbest_target), current_alpha * random.random())
                v_social = scalar_multiply(arbitrary_velocity(p.tour, nbest_target), current_beta * random.random())
                p.velocity = velocity_add(v_inertia, velocity_add(v_cognitive, v_social))
                
                # Particle Movement
                improved = p.move()
                if improved:
                    swarm_improved = True
                
                # Apply 2-opt with SA acceptance occasionally to individual particles
                if random.random() < 0.05:  # 5% chance per particle per iteration
                    optimized_tour, optimized_len = two_opt_local_search(p.tour, d, max_iterations=20)
                    
                    # Accept new tour if better or based on simulated annealing
                    current_len = tour_length(p.tour, d)
                    if accept_with_sa(current_len, optimized_len, temperature):
                        p.tour = optimized_tour
                        # Update personal best if improved
                        if optimized_len < p.best_len:
                            p.best_len = optimized_len
                            p.best_tour = optimized_tour[:]
                            p.stagnation_count = 0
                            swarm_improved = True
            
            # --- Update Swarm Best ---
            if swarm_improved:
                new_swarm_best_tour, new_swarm_best_len = min(((p.best_tour, p.best_len) for p in swarm), key=lambda x: x[1])
                if new_swarm_best_len < swarm_best_lens[swarm_idx]:
                    swarm_best_tours[swarm_idx] = new_swarm_best_tour
                    swarm_best_lens[swarm_idx] = new_swarm_best_len
                    
                    # Update global best if needed
                    if new_swarm_best_len < gbest_len:
                        gbest_tour = new_swarm_best_tour[:]
                        gbest_len = new_swarm_best_len
                        last_improvement_iter = it
                        stagnation_count = 0
                    
        # --- Information Exchange Between Swarms (every 10 iterations) ---
        if it % 10 == 0 and len(swarm_particles) > 1:
            # Share best solutions between swarms by copying to random particles
            for swarm_idx, swarm in enumerate(swarm_particles):
                # Only share if other swarms have better solutions
                better_tours = [(i, tour, length) for i, (tour, length) in 
                               enumerate(zip(swarm_best_tours, swarm_best_lens)) 
                               if i != swarm_idx and length < swarm_best_lens[swarm_idx]]
                
                if better_tours:
                    # Select a random better tour to share
                    other_idx, other_tour, _ = random.choice(better_tours)
                    
                    # Share with a random particle in this swarm
                    target_idx = random.randint(0, len(swarm) - 1)
                    target_particle = swarm[target_idx]
                    
                    # Give the particle a perturbed version of the better tour
                    target_particle.tour = other_tour[:]
                    # Apply a few random swaps to diversify
                    for _ in range(3):
                        i, j = random.sample(range(n), 2)
                        target_particle.tour[i], target_particle.tour[j] = target_particle.tour[j], target_particle.tour[i]
                    target_particle.tour = canonical(target_particle.tour)
        
        # --- Periodic Global 2-opt Local Search on Global Best ---
        if it % local_search_freq == 0:
            optimized_tour, optimized_len = two_opt_local_search(gbest_tour, d, max_iterations=100)
            if optimized_len < gbest_len:
                gbest_tour = optimized_tour
                gbest_len = optimized_len
                last_improvement_iter = it
                stagnation_count = 0
        
        # --- Update Pheromone Matrix ---
        # Evaporation
        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - rho)
        
        # Deposition from global best and all swarm bests
        deposit_amount = q / gbest_len
        for i in range(n):
            u, v = gbest_tour[i], gbest_tour[(i + 1) % n]
            pheromones[u][v] += deposit_amount
            pheromones[v][u] += deposit_amount  # Symmetric for pheromones
        
        # Smaller deposits from swarm bests (diversity)
        for swarm_idx, (swarm_tour, swarm_len) in enumerate(zip(swarm_best_tours, swarm_best_lens)):
            if swarm_tour != gbest_tour:  # Skip if same as global best
                swarm_deposit = 0.5 * q / swarm_len  # Half the deposit amount
                for i in range(n):
                    u, v = swarm_tour[i], swarm_tour[(i + 1) % n]
                    pheromones[u][v] += swarm_deposit
                    pheromones[v][u] += swarm_deposit
        
        # --- Handle Global Stagnation ---
        if it - last_improvement_iter > 200:  # No improvement for 200 iterations
            stagnation_count += 1
            
            # Periodically reinitialize worst-performing particles across all swarms
            if stagnation_count % 5 == 0:  # Every 5 stagnation iterations
                # Sort all particles by best_len
                sorted_particles = sorted(all_particles, key=lambda p: p.best_len, reverse=True)
                
                # Reset worst 20% of particles
                num_to_reset = len(all_particles) // 5
                for i in range(num_to_reset):
                    sorted_particles[i].reset(n, gbest_tour)
                
                # Intensify pheromone for global best path
                boost_amount = 2.0 * q / gbest_len
                for i in range(n):
                    u, v = gbest_tour[i], gbest_tour[(i + 1) % n]
                    pheromones[u][v] += boost_amount
                    pheromones[v][u] += boost_amount
    
    # --- Apply Final Intense Local Search ---
    final_tour, final_len = two_opt_local_search(gbest_tour, d, max_iterations=1000)
    if final_len < gbest_len:
        gbest_tour = final_tour
        gbest_len = final_len
    
    # --- Return Final Best ---
    return gbest_tour, gbest_len


# --- Call the Hybrid Algorithm ---
tour, tour_length = aco_pso_tsp(
    dist_matrix,
    num_particles=num_parts,
    max_iter=max_it,
    theta=theta,
    alpha=alpha,
    beta=beta,
    delta=delta,
    rho=pheromone_rho,
    q=pheromone_q,
    initial_pheromone=pheromone_initial,
    local_search_freq=20  # Apply 2-opt local search every 20 iterations
)

# Update added_note for the enhanced version
added_note = (f"Enhanced ACO-PSO Hybrid with anti-stagnation strategies. "
              f"Base PSO Params: theta={theta}, alpha={alpha}, beta={beta}, delta={'inf' if math.isinf(delta) else delta}. "
              f"ACO Params: rho={pheromone_rho}, q={pheromone_q}, init_ph={pheromone_initial}. "
              f"Anti-stagnation strategies: Multi-swarm with diversified parameters, 2-opt local search, "
              f"simulated annealing acceptance, particle resetting, stagnation detection, adaptive parameters, "
              f"and information exchange between swarms.")






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
