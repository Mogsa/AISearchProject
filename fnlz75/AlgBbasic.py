import os
import sys
import time
import random
from datetime import datetime
import math # Import math for floor function

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
    try:
        with open(input_file, 'r') as the_file:
            # Read the entire file at once for potentially better efficiency
            # This might be memory-intensive for extremely large non-city files,
            # but should be fine for typical TSP files.
            full_content = the_file.read()
        file_string = ""
        ord_range_len = len(ord_range)
        for current_char in full_content:
            for i in range(ord_range_len):
                if ord(current_char) >= ord_range[i][0] and ord(current_char) <= ord_range[i][1]:
                    file_string += current_char
                    break # Found valid range, move to next char
        return file_string
    except FileNotFoundError:
        print(f"*** Error: Input file not found at {input_file}")
        # Or handle differently, maybe return None or raise error
        return None
    except Exception as e:
        print(f"*** Error reading file {input_file}: {e}")
        return None


def remove_all_spaces(the_string):
    # Using string replace method is generally faster than iterating
    return the_string.replace(" ", "")

def integerize(the_string):
    # Improved integerize to handle potential non-digit characters gracefully
    # and avoid issues with leading zeros if the stripped string becomes just "0"
    # It also handles negative signs if they were allowed (they aren't here)
    length = len(the_string)
    stripped_string = ""
    has_digit = False
    for i in range(length):
        if '0' <= the_string[i] <= '9':
            stripped_string += the_string[i]
            has_digit = True
    if not has_digit: # If no digits found
        raise ValueError(f"Cannot convert non-numeric string '{the_string}' to integer.")
    # Let int() handle potential large numbers
    return int(stripped_string)


def convert_to_list_of_int(the_string):
    list_of_integers = []
    location = 0
    finished = False
    the_string_len = len(the_string)

    while not finished and location < the_string_len:
        # Find the next comma or the NOTE= marker, whichever comes first
        found_comma = the_string.find(',', location)
        found_note = the_string.find('NOTE=', location)

        # Determine the end of the current number segment
        end_location = -1
        process_as_last = False

        if found_comma != -1 and found_note != -1:
            # Both comma and NOTE= found, take the earlier one
            if found_comma < found_note:
                end_location = found_comma
            else:
                end_location = found_note
                process_as_last = True # Stop after processing this segment
        elif found_comma != -1:
            # Only comma found
            end_location = found_comma
        elif found_note != -1:
            # Only NOTE= found
            end_location = found_note
            process_as_last = True
        else:
            # Neither found, process the rest of the string as the last number
            end_location = the_string_len
            process_as_last = True

        # Extract the number string
        num_str = the_string[location:end_location].strip()

        if num_str: # Process only if the substring is not empty
            try:
                list_of_integers.append(integerize(num_str))
            except ValueError as e:
                 # Provide more context on error
                 print(f"*** error: Failed to convert segment '{num_str}' to integer while parsing distances.")
                 print(f"   Context: Around location {location} in string '{the_string[:location+50]}...'")
                 raise e # Re-raise the error to stop execution

        # Update location for the next iteration
        if process_as_last:
            finished = True
        else:
             # Check if we stopped at NOTE= or end of string
            if end_location == found_note or end_location == the_string_len:
                 finished = True
            else: # Must have been a comma
                 location = end_location + 1


    return list_of_integers


def build_distance_matrix(num_cities, distances, city_format):
    dist_matrix = []
    i = 0
    # Pre-allocate matrix with zeros for efficiency (using list comprehensions)
    dist_matrix = [[0 for _ in range(num_cities)] for _ in range(num_cities)]

    if city_format == "full":
        if len(distances) != num_cities * num_cities:
             print(f"*** error: Expected {num_cities*num_cities} distances for full matrix, found {len(distances)}.")
             sys.exit()
        idx = 0
        for r in range(num_cities):
            for c in range(num_cities):
                dist_matrix[r][c] = distances[idx]
                idx += 1
    elif city_format == "upper_tri":
        if len(distances) != (num_cities * (num_cities + 1)) // 2:
             print(f"*** error: Expected {(num_cities * (num_cities + 1)) // 2} distances for upper_tri matrix, found {len(distances)}.")
             sys.exit()
        idx = 0
        for r in range(num_cities):
            for c in range(r, num_cities): # Fill upper triangle including diagonal
                value = distances[idx]
                dist_matrix[r][c] = value
                if r != c: # Fill lower triangle symmetrically
                    dist_matrix[c][r] = value
                idx += 1
    elif city_format == "strict_upper_tri":
         if len(distances) != (num_cities * (num_cities - 1)) // 2:
              print(f"*** error: Expected {(num_cities * (num_cities - 1)) // 2} distances for strict_upper_tri matrix, found {len(distances)}.")
              sys.exit()
         idx = 0
         for r in range(num_cities):
             for c in range(r + 1, num_cities): # Fill strict upper triangle
                 value = distances[idx]
                 dist_matrix[r][c] = value
                 dist_matrix[c][r] = value # Fill lower triangle symmetrically
                 idx += 1
    else: # Should not happen if previous checks worked
         print(f"*** error: Unknown city format '{city_format}'.")
         sys.exit()

    return dist_matrix


def read_in_algorithm_codes_and_tariffs(alg_codes_file):
    flag = "good"
    code_dictionary = {}
    tariff_dictionary = {}
    if not os.path.exists(alg_codes_file):
        flag = "not_exist"
        return code_dictionary, tariff_dictionary, flag
    try:
        ord_range = [[32, 126]]
        file_string = read_file_into_string(alg_codes_file, ord_range)
        if file_string is None: # Handle read error from helper
             flag = "read_error"
             return code_dictionary, tariff_dictionary, flag

        location = 0
        EOF = False
        list_of_items = []
        while not EOF:
            found_comma = file_string.find(",", location)
            if found_comma == -1:
                EOF = True
                sandwich = file_string[location:].strip() # Strip whitespace
            else:
                sandwich = file_string[location:found_comma].strip()
                location = found_comma + 1
            # Append only if sandwich is not empty
            if sandwich:
                 list_of_items.append(sandwich)

        # Check if the number of items is a multiple of 3
        if not list_of_items or len(list_of_items) % 3 != 0:
             print(f"*** error: Parsing 'alg_codes_and_tariffs.txt' failed. Expected multiple of 3 non-empty items, found {len(list_of_items)}.")
             # print(f"   Items found: {list_of_items}") # Debug print
             flag = "bad_format"
             # Return empty dicts but set flag
             return {}, {}, flag

        third_length = len(list_of_items) // 3
        for i in range(third_length):
            code = list_of_items[3 * i]
            name = list_of_items[3 * i + 1]
            tariff_str = list_of_items[3 * i + 2]
            try:
                tariff = int(tariff_str)
                code_dictionary[code] = name
                tariff_dictionary[code] = tariff
            except ValueError:
                print(f"*** error: Invalid tariff value '{tariff_str}' for code '{code}' in 'alg_codes_and_tariffs.txt'. Must be an integer.")
                flag = "bad_format"
                # Optionally, return immediately or continue processing other entries
                # return code_dictionary, tariff_dictionary, flag
        if flag == "bad_format": # Check if error occurred inside loop
             return {}, {}, flag

    except Exception as e:
        print(f"*** error: An unexpected error occurred while reading '{alg_codes_file}': {e}")
        flag = "read_error"

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

input_file = "AISearchfile058.txt" # CHANGE THIS OR SUPPLY VIA COMMAND LINE

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
# --- Read City File ---
path_to_input_file = os.path.join(path_for_city_files, input_file)
if os.path.isfile(path_to_input_file):
    try:
        ord_range = [[32, 126]]
        file_string = read_file_into_string(path_to_input_file, ord_range)
        if file_string is None: sys.exit() # Error handled in function
        file_string = remove_all_spaces(file_string)
        print("I have found and read the input file " + input_file + ":")
    except Exception as e:
        print(f"*** error: Failed to read city file '{input_file}'. Error: {e}")
        sys.exit()
else:
    print(f"*** error: The city file '{input_file}' does not exist in the folder '{path_for_city_files}'.")
    sys.exit()

# --- Parse City File ---
location = file_string.find("SIZE=")
if location == -1:
    print(f"*** error: SIZE= not found in the city file '{input_file}'. Incorrect format.")
    sys.exit()

comma = file_string.find(",", location)
if comma == -1:
    print(f"*** error: Comma missing after SIZE= in the city file '{input_file}'. Incorrect format.")
    sys.exit()

num_cities_as_string = file_string[location + 5:comma]
try:
    num_cities = integerize(num_cities_as_string)
    if num_cities < 0: # Allow 0 cities for empty problem
         raise ValueError("Number of cities cannot be negative.")
    print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))
except ValueError as e:
     print(f"*** error: Invalid number of cities '{num_cities_as_string}' in '{input_file}'. {e}")
     sys.exit()


comma = comma + 1
stripped_file_string = file_string[comma:]
try:
    distances = convert_to_list_of_int(stripped_file_string)
except Exception as e:
     print(f"*** error: Failed to parse distances in '{input_file}'. Check format. Error: {e}")
     sys.exit()


# --- Determine distance format and build matrix ---
counted_distances = len(distances)
expected_full = num_cities * num_cities
expected_upper_tri = (num_cities * (num_cities + 1)) // 2
expected_strict_upper_tri = (num_cities * (num_cities - 1)) // 2

city_format = None # Initialize
if num_cities == 0:
     if counted_distances == 0:
          city_format = "empty" # Handle 0 cities case
     else:
          print(f"*** error: Non-zero number of distances ({counted_distances}) found for zero cities.")
          sys.exit()
elif counted_distances == expected_full:
    city_format = "full"
elif counted_distances == expected_upper_tri:
    city_format = "upper_tri"
elif counted_distances == expected_strict_upper_tri:
    city_format = "strict_upper_tri"
else:
    print(f"*** error: Incorrect number of distances in '{input_file}'.")
    print(f"   Found {counted_distances}, expected {expected_full} (full), {expected_upper_tri} (upper_tri), or {expected_strict_upper_tri} (strict_upper_tri) for {num_cities} cities.")
    sys.exit()

dist_matrix = [] # Initialize
if city_format != "empty":
     try:
         dist_matrix = build_distance_matrix(num_cities, distances, city_format)
         print("   the distance matrix 'dist_matrix' has been built.")
     except IndexError:
          print(f"*** error: Index out of bounds while building distance matrix from '{input_file}'. Check distance list length and format.")
          sys.exit()
     except Exception as e:
          print(f"*** error: An unexpected error occurred building distance matrix: {e}")
          sys.exit()
else:
     print("   num_cities is 0, distance matrix is empty.")


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

if flag == "not_exist":
    print(f"*** error: The text file 'alg_codes_and_tariffs.txt' does not exist at '{path_for_alg_codes_and_tariffs}'.")
    sys.exit()
elif flag != "good":
     print(f"*** error: Failed to read or parse 'alg_codes_and_tariffs.txt' correctly (flag: {flag}).")
     # Exit might depend on whether the dictionaries are essential immediately
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

my_user_name = "fnlz75" # Change this line

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

my_first_name = ""
my_last_name = ""

############ START OF SECTOR 7 (IGNORE THIS COMMENT)
############
############ YOU NEED TO SUPPLY THE ALGORITHM CODE IN THE RESERVED STRING VARIABLE 'algorithm_code'
############ FOR THE ALGORITHM YOU ARE IMPLEMENTING. IT NEEDS TO BE A LEGAL CODE FROM THE TEXT-FILE
############ 'alg_codes_and_tariffs.txt' (READ THIS FILE TO SEE THE CODES).
############
############ END OF SECTOR 7 (IGNORE THIS COMMENT)

algorithm_code = "PS" # Set algorithm code to Particle Swarm

############ START OF SECTOR 8 (IGNORE THIS COMMENT)
############
############ PLEASE SCROLL DOWN UNTIL THE NEXT BLOCK OF CAPITALIZED COMMENTS STARTING
############ 'HAVE YOU TOUCHED ...'
############
############ DO NOT TOUCH OR ALTER THE CODE IN BETWEEN! YOU HAVE BEEN WARNED!
############

# Check algorithm code validity *before* starting timer
if not algorithm_code in code_dictionary:
    print(f"*** error: the algorithm code '{algorithm_code}' is illegal. Check 'alg_codes_and_tariffs.txt'.")
    sys.exit()
# Check if tariff exists for the code, needed later by skeleton potentially
if not algorithm_code in tariff_dictionary:
     print(f"*** error: tariff not found for algorithm code '{algorithm_code}' in 'alg_codes_and_tariffs.txt'.")
     sys.exit() # Tariff might be used by validation/grading scripts

print(f"   your algorithm code is legal and is {algorithm_code} - {code_dictionary.get(algorithm_code, 'Unknown Algorithm Name')}.") # Use .get for safety

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

added_note = "" # You can add notes here if you wish

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

############ START OF SECTOR 9 (IGNORE THIS COMMENT)


# ------------- parameters -------------
num_particles = max(10, min(60, num_cities // 3))  # scale with problem size
max_iter      = 2500                               # more loops – each is cheap
w_start, w_end = 0.9, 0.4                          # inertia damp
c1, c2         = 1.2, 1.6                         # cognitive / social
w_max, w_min   = w_start, w_end                    # names expected by wrapper
# --------------------------------------

import random, math
random.seed()

# ---- helpers -----------------------------------------------------

def calculate_tour_length(tour, matrix):
    length = 0
    n = len(tour)
    for i in range(n):
        length += matrix[tour[i]][tour[(i + 1) % n]]
    return length


def keys_to_tour(keys):
    """decode random‑keys vector to permutation"""
    return [i for i, _ in sorted(enumerate(keys), key=lambda t: t[1])]


def tour_length(tour):
    return calculate_tour_length(tour, dist_matrix)


# ---- greedy seeding ---------------------------------------------

def greedy_keys(start):
    """Return random‑keys vector that decodes to a greedy NN tour."""
    visited = {start}
    tour = [start]
    while len(tour) < num_cities:
        last = tour[-1]
        nxt = min((c for c in range(num_cities) if c not in visited),
                   key=lambda c: dist_matrix[last][c])
        visited.add(nxt)
        tour.append(nxt)
    rk = [0] * num_cities
    for rank, city in enumerate(tour):
        rk[city] = rank / (num_cities - 1)
    return rk


# ---- 2‑opt (first improvement) ----------------------------------

def two_opt_first(tour):
    best = tour_length(tour)
    n = len(tour)
    for i in range(n - 2):
        for j in range(i + 2, n - (i == 0)):
            a, b = tour[i], tour[(i + 1) % n]
            c, d = tour[j], tour[(j + 1) % n]
            delta = (dist_matrix[a][c] + dist_matrix[b][d]
                     - dist_matrix[a][b] - dist_matrix[c][d])
            if delta < 0:
                tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                best += delta
                return tour, best
    return tour, best


# ---- initialise swarm -------------------------------------------

particles = []
for _ in range(num_particles):
    if random.random() < 0.30:          # 30 % smart particles
        pos = greedy_keys(random.randrange(num_cities))
    else:
        pos = [random.random() for _ in range(num_cities)]
    vel  = [0.0] * num_cities
    tour = keys_to_tour(pos)
    fit  = tour_length(tour)
    particles.append({"pos": pos, "vel": vel,
                      "pbest_pos": pos[:], "pbest_fit": fit,
                      "tour": tour, "fit": fit})

# ---- global best -------------------------------------------------

g_best_particle = min(particles, key=lambda p: p["fit"])
g_best_pos  = g_best_particle["pbest_pos"][:]
g_best_fit  = g_best_particle["fit"]
g_best_tour = keys_to_tour(g_best_pos)

# ---- main PSO loop ----------------------------------------------

no_improve = 0
best_so_far = g_best_fit

for it in range(max_iter):
    w = w_start - (w_start - w_end) * it / max_iter
    for p in particles:
        # --- velocity & position update ---
        for d in range(num_cities):
            r1, r2 = random.random(), random.random()
            p["vel"][d] = ( w * p["vel"][d]
                             + c1 * r1 * (p["pbest_pos"][d] - p["pos"][d])
                             + c2 * r2 * (g_best_pos[d] - p["pos"][d]) )
            new_pos = p["pos"][d] + p["vel"][d]
            # reflecting bounds
            if new_pos < 0.0:
                p["pos"][d] = -new_pos
                p["vel"][d] = -p["vel"][d] * 0.5
            elif new_pos > 1.0:
                p["pos"][d] = 2.0 - new_pos
                p["vel"][d] = -p["vel"][d] * 0.5
            else:
                p["pos"][d] = new_pos

        # --- fitness evaluation ---
        p["tour"] = keys_to_tour(p["pos"])
        p["fit"]  = tour_length(p["tour"])

        # --- personal best update + 2‑opt kick ---
        if p["fit"] < p["pbest_fit"]:
            improved_tour, imp_fit = two_opt_first(p["tour"][:])
            if imp_fit < p["pbest_fit"]:
                new_keys = [0] * num_cities
                for rank, city in enumerate(improved_tour):
                    new_keys[city] = rank / (num_cities - 1)
                p["pbest_pos"], p["pbest_fit"] = new_keys, imp_fit
            else:
                p["pbest_pos"], p["pbest_fit"] = p["pos"][:], p["fit"]

        # --- global best update ---
        if p["pbest_fit"] < g_best_fit:
            g_best_fit  = p["pbest_fit"]
            g_best_pos  = p["pbest_pos"][:]
            g_best_tour = keys_to_tour(g_best_pos)

    # ---- stagnation check & jolt --------------------------------
    if g_best_fit < best_so_far - 1e-6:
        best_so_far = g_best_fit
        no_improve  = 0
    else:
        no_improve += 1

    if no_improve >= 50:
        worst = sorted(particles, key=lambda p: p["fit"], reverse=True)
        for p in worst[:max(1, num_particles // 5)]:
            p["pos"] = [random.random() for _ in range(num_cities)]
            p["vel"] = [0.0] * num_cities
            p["tour"] = keys_to_tour(p["pos"])
            p["fit"]  = tour_length(p["tour"])
            p["pbest_pos"], p["pbest_fit"] = p["pos"][:], p["fit"]
        # reset global best
        g_best_particle = min(particles, key=lambda p: p["fit"])
        g_best_pos  = g_best_particle["pbest_pos"][:]
        g_best_fit  = g_best_particle["fit"]
        g_best_tour = keys_to_tour(g_best_pos)
        best_so_far = g_best_fit
        no_improve  = 0

# ---- hand results to the wrapper --------------------------------

tour        = g_best_tour
tour_length = int(round(g_best_fit))

max_it    = max_iter
num_parts = num_particles
added_note = ("Improved PSO with greedy seeding, 2‑opt, reflecting bounds, "
              f"restarts | N={num_particles}, iter={max_iter}, "
              f"w∈[{w_start},{w_end}], c1={c1}, c2={c2}")


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

# --- The following code validates the tour and writes the output file ---
# --- Slightly modified validation from previous steps to be more robust ---

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
    # Include other relevant params in note if they exist
    ps_params = f"'max_it' = {max_it}, 'num_parts' = {num_parts}"
    # Add others if defined and needed for record
    if 'w_max' in locals() and 'w_min' in locals(): ps_params += f", w=[{w_max}-{w_min:.2f}]"
    if 'c1' in locals(): ps_params += f", c1={c1}"
    if 'c2' in locals(): ps_params += f", c2={c2}"
    if 'max_stagnation' in locals(): ps_params += f", stag={max_stagnation}"
    added_note = added_note + f"The parameter values are {ps_params}."


added_note = added_note + "\nRUN-TIME = " + str(elapsed_time) + " seconds.\n"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
added_note = added_note + "DATE-TIME = " + dt_string + ".\n"

# --- Basic validation checks ---
if 'tour' not in locals() or 'tour_length' not in locals():
     print("*** error: 'tour' or 'tour_length' not defined by the algorithm.")
     sys.exit()

flag = "good"
# Ensure tour is a list
if not isinstance(tour, list):
    flag = "bad_type"
    print("*** error: Your final 'tour' is not a list.")
    sys.exit()
else:
    length = len(tour)
    # Handle empty tour case gracefully in checks
    if length == 0 and num_cities == 0:
         pass # Empty tour for 0 cities is valid
    elif length == 0 and num_cities > 0:
         print("*** error: Your final 'tour' is empty, but num_cities > 0.")
         sys.exit()
    elif length > 0: # Tour is not empty, check elements
        for i in range(length):
            # Ensure elements are integers
            if not isinstance(tour[i], int):
                 try:
                    tour[i] = int(round(tour[i]))
                 except (ValueError, TypeError):
                     flag = "bad_element_type"
                     break
        if flag == "bad_element_type":
             print(f"*** error: Your tour contains non-integer values (e.g., at index {i}, value {tour[i]}) that cannot be converted.")
             sys.exit()

# Ensure tour_length is an integer (allow sys.maxsize)
if not isinstance(tour_length, int) and tour_length != sys.maxsize:
     try:
         tour_length = int(round(tour_length))
     except (ValueError, TypeError):
         print(f"*** error: The tour-length '{tour_length}' is not an integer value and cannot be converted.")
         sys.exit()


# Check tour length against num_cities (only if num_cities > 0)
if num_cities > 0 and len(tour) != num_cities:
    print(f"*** error: The tour does not consist of {num_cities} cities as there are, in fact, {len(tour)}.")
    sys.exit()

# Check if all cities 0 to num_cities-1 are present exactly once (only if num_cities > 0)
if num_cities > 0:
    if not tour: # Should be caught above, but double-check
         print("*** error: Tour is empty during city validation step.")
         sys.exit()

    city_counts = {}
    flag = "good" # Reset flag for this check
    for city in tour:
        if not isinstance(city, int) or not (0 <= city < num_cities): # Check type and range
            flag = "bad_city_value"
            invalid_city_value = city # Store the problematic value
            break # Exit loop

        city_counts[city] = city_counts.get(city, 0) + 1


    if flag == "bad_city_value":
        print(f"*** error: Your tour contains an invalid city index or type: '{invalid_city_value}' (valid range 0 to {num_cities-1}, integer type).")
        sys.exit()

    # Check for missing or duplicate cities using counts
    missing_cities = [c for c in range(num_cities) if c not in city_counts]
    duplicate_cities = [c for c, count in city_counts.items() if count > 1]

    if missing_cities:
         print(f"*** error: Your tour is missing cities: {missing_cities}.")
         flag = "bad"
    if duplicate_cities:
         print(f"*** error: Your tour contains duplicate cities: {duplicate_cities}.")
         flag = "bad"

    if flag == "bad":
         sys.exit() # Exit if missing or duplicates found

# --- Recalculate tour length for verification ---
# Skip check if reported length is maxsize or num_cities < 2
if tour_length == sys.maxsize:
     check_tour_length = sys.maxsize
     # print("   Skipping length verification as reported length is maxsize.")
elif num_cities < 2:
     check_tour_length = 0 # Length is 0 for 0 or 1 city
     if tour_length != 0:
           print(f"*** warning: Reported tour length is {tour_length} but should be 0 for {num_cities} cities.")
           # Allow proceeding but flag potential issue
elif not tour: # Should be caught above
     check_tour_length = sys.maxsize
     print("*** Warning: Attempting to check length of an empty tour.")
else:
     # Use the helper function for consistency
     check_tour_length = calculate_tour_length(tour, dist_matrix)
     if check_tour_length == float('inf'):
          check_tour_length = sys.maxsize # Use consistent large number


# Final comparison (only if check_tour_length is valid)
if check_tour_length != sys.maxsize and tour_length != sys.maxsize and abs(tour_length - check_tour_length) > 1e-6 :
    original_gbest_fit_str = f"{gbest_fit:.4f}" if 'gbest_fit' in locals() and isinstance(gbest_fit, float) else str(locals().get('gbest_fit', 'N/A'))
    print(f"*** error: The reported tour_length ({tour_length}, possibly rounded from {original_gbest_fit_str}) does not match the recalculated length ({check_tour_length}) for the provided tour.")
    # print(f"Tour provided: {tour}") # Optional: print tour for debugging
    sys.exit()
elif check_tour_length == sys.maxsize and tour_length != sys.maxsize:
     print(f"*** error: Reported tour length is {tour_length}, but recalculation indicates an invalid tour (check distance matrix or tour indices).")
     sys.exit()


# --- Success Message ---
# Handle maxsize output gracefully
if tour_length == sys.maxsize:
     print(f"You, user {my_user_name}, have completed the run, but the best tour found has an invalid (infinite or maxsize) length.")
else:
     print(f"You, user {my_user_name}, have successfully built a tour of length {tour_length}!")

# --- Certificate calculation and file naming ---
len_user_name = len(my_user_name)
user_number = 0
for i in range(0, len_user_name):
    user_number = user_number + ord(my_user_name[i])
alg_number = 0
if algorithm_code and len(algorithm_code) >= 2: # Basic check
    alg_number = ord(algorithm_code[0]) + ord(algorithm_code[1])
len_dt_string = len(dt_string)
date_time_number = 0
for i in range(0, len_dt_string):
    date_time_number = date_time_number + ord(dt_string[i])
tour_diff = 0
# Calculate tour_diff only if tour is valid and has > 1 city
if tour and num_cities > 1:
    try:
         # Use modulo for wrap-around difference
         tour_diff = abs(tour[0] - tour[-1]) # Diff between first and last
         for i in range(num_cities - 1):
             tour_diff += abs(tour[i + 1] - tour[i]) # Diff between consecutive
    except IndexError:
         print("*** Warning: Index error during certificate calculation. Check tour validity.")
         tour_diff = 0 # Default or handle error
certificate = user_number + alg_number + date_time_number + tour_diff
local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]
# Ensure script_name and input_file names are reasonable before creating filename
safe_script_name = "".join(c for c in script_name if c.isalnum() or c in ('_', '-')).replace('.py','')
safe_input_file = "".join(c for c in input_file if c.isalnum() or c in ('_', '-')).replace('.txt','')
output_file_name = f"{safe_script_name}_{safe_input_file}_{output_file_time}.txt"


# --- Write output file ---
try:
    with open(output_file_name,'w') as f: # Use 'with' for safer file handling
        f.write(f"USER = {my_user_name} ({my_first_name} {my_last_name}),\n")
        f.write(f"ALGORITHM CODE = {algorithm_code}, NAME OF CITY-FILE = {input_file},\n")
        f.write(f"SIZE = {num_cities}, TOUR LENGTH = {tour_length},\n")
        # Write the tour
        if tour: # Only write tour if it exists and is not empty
            f.write(str(tour[0]))
            for i in range(1,len(tour)): # Iterate up to actual length
                f.write(f",{tour[i]}")
        f.write(",\nNOTE = {0}".format(added_note)) # Write note always
        f.write(f"CERTIFICATE = {certificate}.\n")
    print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")
except IOError as e:
    print(f"*** error: Could not write tour file '{output_file_name}'. Check permissions. Error: {e}")
    sys.exit()
except Exception as e:
    print(f"*** error: An unexpected error occurred while writing the tour file: {e}")
    sys.exit()


############ END OF SECTOR 10 (IGNORE THIS COMMENT)