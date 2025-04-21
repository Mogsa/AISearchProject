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


Tour     = List[int]
Swap     = Tuple[int, int]      # arbitrary transposition (city_i, city_j)
Velocity = List[Swap]
Particles = List["Particle"]

# ---------------------------- helper functions ----------------------------- #

def canonical(tour: Tour) -> Tour:
    """Rotate so city 0 is first; return *copy*."""
    k = tour.index(0)
    return tour[k:] + tour[:k]


def tour_length(tour: Tour, d) -> int:
    return sum(d[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


# ---- unique velocity (arbitrary swaps) ------------------------------------ #

def permutation_from_tours(a: Tour, b: Tour) -> Dict[int, int]:
    """Return permutation ρ (as dict city→city) such that applying ρ to a yields b."""
    return {a[i]: b[i] for i in range(len(a))}


def cycle_decomposition(perm: Dict[int, int]) -> List[List[int]]:
    """Return list of cycles (each list of cities). Ignores 1‑cycles."""
    cycles = []
    seen = set()
    for x in perm:
        if x in seen or perm[x] == x:
            continue
        cycle = [x]
        y = perm[x]
        while y != x:
            cycle.append(y)
            y = perm[y]
        seen.update(cycle)
        cycles.append(cycle)
    # order by decreasing length then lowest city id
    cycles.sort(key=lambda c: (-len(c), min(c)))
    return cycles


def velocity_from_cycles(cycles: List[List[int]]) -> Velocity:
    v: Velocity = []
    for cyc in cycles:
        # convert (c0 c1 ... ck-1) to (c0 c1)(c1 c2)...
        for i in range(len(cyc) - 1):
            v.append((cyc[i], cyc[i + 1]))
    return v


def arbitrary_velocity(a: Tour, b: Tour) -> Velocity:
    perm = permutation_from_tours(a, b)
    cycles = cycle_decomposition(perm)
    return velocity_from_cycles(cycles)


def two_opt_swap(tour: Tour, i: int, j: int) -> Tour:
    """Create a new tour by reversing the segment from position i to j."""
    if i > j:
        i, j = j, i
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]


def two_opt_local_search(tour: Tour, d, max_iterations: int = 100) -> Tuple[Tour, int]:
    """Apply 2-opt local search to improve a tour."""
    best_tour = tour[:]
    best_length = tour_length(best_tour, d)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(1, len(tour)-2):  # Skip city 0 to maintain canonical form
            for j in range(i+1, len(tour)):
                if j - i == 1:  # Skip adjacent cities
                    continue
                    
                # Apply 2-opt swap
                new_tour = two_opt_swap(best_tour, i, j)
                new_length = tour_length(new_tour, d)
                
                if new_length < best_length:
                    best_tour = new_tour
                    best_length = new_length
                    improved = True
                    break  # Restart with the improved tour
                    
            if improved:
                break
    
    return best_tour, best_length


# ---- velocity algebra ----------------------------------------------------- #

def apply_velocity(tour: Tour, v: Velocity) -> Tour:
    tour = tour[:]
    pos = {city: idx for idx, city in enumerate(tour)}  # city→index map for O(1) swaps
    for c1, c2 in v:
        i, j = pos[c1], pos[c2]
        tour[i], tour[j] = tour[j], tour[i]
        pos[c1], pos[c2] = j, i
    return tour


def velocity_add(v1: Velocity, v2: Velocity) -> Velocity:
    return v1 + v2


def scalar_multiply(v: Velocity, gamma: float) -> Velocity:
    if gamma <= 0 or not v:
        return []
    if gamma < 1:
        k = int(math.floor(gamma * len(v)))
        return v[:k]
    k = int(math.floor(gamma))
    frac = gamma - k
    return v * k + scalar_multiply(v, frac)

# ---------------------------- Particle class ------------------------------- #
class Particle:
    def __init__(self, init_tour: Tour, d):
        self.d = d
        self.tour: Tour = canonical(init_tour)
        self.velocity: Velocity = []
        self.best_tour: Tour = self.tour[:]
        self.best_len  = tour_length(self.tour, d)

    def move(self):
        self.tour = apply_velocity(self.tour, self.velocity)
        L = tour_length(self.tour, self.d)
        if L < self.best_len:
            self.best_len = L
            self.best_tour = self.tour[:]

# ---------------------------- PSO routine ---------------------------------- #

def pso_tsp(
    d,
    *,
    num_particles: int = 40,
    max_iter: int = 1000,
    theta: float = 0.8,
    alpha: float = 1.5,
    beta: float = 1.5,
    delta: int | float = float("inf"),
    time_limit: float | None = None,
    seed: int | None = None,
):
    """Discrete PSO with *arbitrary‑swap* velocities. Returns (best_tour, length)."""
    if seed is not None:
        random.seed(seed)

    n = len(d)

    # --- initialise swarm ------------------------------------------------- #
    particles: Particles = []
    for _ in range(num_particles):
        t = list(range(n))
        random.shuffle(t)
        t = canonical(t)
        p = Particle(t, d)
        # random velocity: m random city‑swaps
        m = random.randint(1, n)  # one swap per particle expected
        vel: Velocity = []
        for _ in range(m):
            c1, c2 = random.sample(range(1, n), 2)  # don’t swap city 0
            vel.append((t[c1], t[c2]))  # store cities not indices
        p.velocity = vel
        particles.append(p)

    gbest_tour, gbest_len = min(((p.best_tour, p.best_len) for p in particles), key=lambda x: x[1])

    start = time.time()
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start) >= time_limit:
            break
        for idx, p in enumerate(particles):
            # neighbourhood best (ring if delta finite)
            if math.isinf(delta):
                nbest_tour, nbest_len = gbest_tour, gbest_len
            else:
                ring = [(idx + i) % num_particles for i in range(-delta, delta + 1)]
                nbest_tour, nbest_len = min(((particles[j].best_tour, particles[j].best_len) for j in ring), key=lambda x: x[1])

            # velocity update ------------------------------------------------#
            v_inertia   = scalar_multiply(p.velocity, theta)
            v_cognitive = scalar_multiply(arbitrary_velocity(p.tour, p.best_tour), alpha * random.random())
            v_social    = scalar_multiply(arbitrary_velocity(p.tour, nbest_tour),  beta  * random.random())
            p.velocity = velocity_add(v_inertia, velocity_add(v_cognitive, v_social))
            p.move()

        # update global best --------------------------------------------------#
        best_p = min(particles, key=lambda q: q.best_len)
        if best_p.best_len < gbest_len:
            gbest_len  = best_p.best_len
            gbest_tour = best_p.best_tour[:]
            
        # Apply 2-opt local search to best solution periodically
        if it % 10 == 0:  # Every 10 iterations
            improved_tour, improved_length = two_opt_local_search(gbest_tour, d, max_iterations=20)
            if improved_length < gbest_len:
                gbest_len = improved_length
                gbest_tour = improved_tour
                
                # Influence particles with improved solution
                for p in particles:
                    # 20% chance to update a particle's velocity towards the improved tour
                    if random.random() < 0.2:
                        extra_social = scalar_multiply(
                            arbitrary_velocity(p.tour, gbest_tour),
                            2.0 * random.random()  # Strong influence
                        )
                        p.velocity = velocity_add(p.velocity, extra_social)
    
    # Apply final intensive local search
    final_tour, final_length = two_opt_local_search(gbest_tour, d, max_iterations=50)
    if final_length < gbest_len:
        gbest_tour, gbest_len = final_tour, final_length
        
    return gbest_tour, gbest_len


# Define our PSO parameters
max_it = 1000
num_parts = num_cities * 3  # Number of particles scales with city count
theta = 0.6    # Inertia weight
alpha = 1.5    # Cognitive parameter (personal best)
beta = 1.5    # Social parameter (global best)
delta = float("inf")  # Global topology (use all particles)

# Run PSO algorithm
tour, tour_length = pso_tsp(dist_matrix, 
                           num_particles=num_parts,
                           max_iter=max_it,
                           theta=theta,
                           alpha=alpha,
                           beta=beta,
                           delta=delta)

# Add extra parameter information
added_note = added_note + f"PSO with 2-opt local search. Theta = {theta}, alpha = {alpha}, beta = {beta}, delta = {'infinite' if math.isinf(delta) else delta}. 2-opt is applied every 10 iterations and at the end."










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
