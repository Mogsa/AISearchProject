import os
import sys
import time
import random
import math # Added for calculations
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
    # Handle potential empty string after stripping non-digits
    if not stripped_string or stripped_string == "0": # Check if result is just "0" potentially from empty input
         # Need to distinguish between actual zero and conversion default/error
         is_actually_zero = False
         original_stripped = "".join(c for c in the_string if '0' <= c <= '9')
         if original_stripped == "0":
             is_actually_zero = True

         if not is_actually_zero and stripped_string == "0": # If it became "0" but wasn't originally
             # This indicates the original might have been empty or non-numeric
             # Returning 0 might be okay, or signal an error/default
             # Let's keep returning 0 for simplicity based on original skeleton logic
             # print(f"Warning: Input '{the_string}' resulted in integer 0 after stripping non-digits.")
             pass # Keep behavior as is, returning 0

    # Attempt conversion, handle errors
    try:
        resulting_int = int(stripped_string)
    except ValueError:
        print(f"Warning: Could not convert '{stripped_string}' (from '{the_string}') to integer, returning 0.")
        return 0
    return resulting_int


def convert_to_list_of_int(the_string):
    list_of_integers = []
    location = 0
    finished = False
    while finished == False:
        found_comma = the_string.find(',', location)
        if found_comma == -1:
            # Process the remaining part of the string after the last comma
            last_part = the_string[location:]
            note_marker = "NOTE="
            note_index = last_part.find(note_marker)
            if note_index != -1:
                num_str = last_part[:note_index]
            else:
                num_str = last_part

            if num_str.strip(): # Check if there's anything to convert
                 list_of_integers.append(integerize(num_str))
            finished = True
        else:
            current_num_str = the_string[location:found_comma]
            if current_num_str.strip(): # Ensure not empty between commas
                list_of_integers.append(integerize(current_num_str))
            location = found_comma + 1
            # Check if the next part starts with "NOTE=" immediately after comma
            if the_string[location:].startswith("NOTE="):
                finished = True

    return list_of_integers


def build_distance_matrix(num_cities, distances, city_format):
    dist_matrix = [[0] * num_cities for _ in range(num_cities)] # Pre-allocate with zeros
    i = 0
    try:
        if city_format == "full":
            if len(distances) != num_cities * num_cities:
                raise ValueError(f"Expected {num_cities*num_cities} distances for 'full' format, got {len(distances)}.")
            for r in range(num_cities):
                for c in range(num_cities):
                    dist_matrix[r][c] = distances[i]
                    i += 1
        elif city_format == "upper_tri":
            expected_len = (num_cities * (num_cities + 1)) // 2
            if len(distances) != expected_len:
                raise ValueError(f"Expected {expected_len} distances for 'upper_tri' format, got {len(distances)}.")
            for r in range(num_cities):
                for c in range(r, num_cities): # Fill diagonal and upper triangle
                    dist_matrix[r][c] = distances[i]
                    i += 1
        elif city_format == "strict_upper_tri":
            expected_len = (num_cities * (num_cities - 1)) // 2
            if len(distances) != expected_len:
                raise ValueError(f"Expected {expected_len} distances for 'strict_upper_tri' format, got {len(distances)}.")
            for r in range(num_cities):
                for c in range(r + 1, num_cities): # Fill only strict upper triangle
                    dist_matrix[r][c] = distances[i]
                    i += 1
        else:
            raise ValueError(f"Unknown city format '{city_format}'.")

        # Symmetrize for triangular formats
        if city_format == "upper_tri" or city_format == "strict_upper_tri":
            for r in range(num_cities):
                for c in range(r + 1, num_cities):
                    dist_matrix[c][r] = dist_matrix[r][c]

    except IndexError:
        print(f"*** error: Index out of bounds while reading distances for format '{city_format}'. Check distance list length.")
        sys.exit()
    except ValueError as e:
        print(f"*** error: {e}")
        sys.exit()

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
        # Handle potential empty strings between commas
        if sandwich.strip():
             list_of_items.append(sandwich.strip())

    # Ensure we have groups of 3 items
    if len(list_of_items) % 3 != 0:
         print(f"*** error: Parsing issue in {alg_codes_file}. Expected items in multiples of 3, found {len(list_of_items)}.")
         # For robustness, try processing full groups of 3 found
         num_complete_groups = len(list_of_items) // 3
         list_of_items = list_of_items[:num_complete_groups * 3]
         if not list_of_items: # If no complete groups, return empty
              return code_dictionary, tariff_dictionary, "format_error"
         flag = "format_error" # Set flag but continue processing

    third_length = len(list_of_items) // 3
    for i in range(third_length):
        code = list_of_items[3 * i]
        desc = list_of_items[3 * i + 1]
        try:
            tariff = int(list_of_items[3 * i + 2])
            code_dictionary[code] = desc
            tariff_dictionary[code] = tariff
        except ValueError:
            print(f"*** error: Invalid tariff value '{list_of_items[3 * i + 2]}' for code '{code}' in {alg_codes_file}.")
            # Optionally skip this entry or assign a default tariff
            flag = "format_error" # Indicate there was an issue

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
# Correct path assumption: script is in 'username/' folder, 'city-files' is sibling to 'username/'
path_for_city_files = os.path.join("..", "city-files")
# Construct the full path to the city file
path_to_input_file = os.path.join(path_for_city_files, input_file)
# Verify the path exists relative to the script's parent directory
if not os.path.exists(os.path.dirname(path_to_input_file)):
     # If the city-files directory isn't where expected, maybe it's in the same directory as the script?
     print(f"Warning: '{path_for_city_files}' not found relative to script's parent directory.")
     # Try path relative to script's current directory instead
     path_for_city_files = "city-files" # Assume it's a subdirectory
     path_to_input_file = os.path.join(path_for_city_files, input_file)
     if not os.path.exists(os.path.dirname(path_to_input_file)):
           # If still not found, fallback or error
           print(f"Warning: Also could not find '{path_for_city_files}' relative to script's directory.")
           # Fallback: assume city file is in the same directory as the script
           path_to_input_file = input_file

############ END OF SECTOR 2 (IGNORE THIS COMMENT)


############ START OF SECTOR 3 (IGNORE THIS COMMENT)
# Ensure path_to_input_file is defined before checking if it's a file
if 'path_to_input_file' not in locals() or not os.path.isfile(path_to_input_file):
     # Try finding the file in common locations if the initial path failed
     found_file = False
     # 1. Relative to script parent, in 'city-files' (original assumption)
     alt_path1 = os.path.join("..", "city-files", input_file)
     if os.path.isfile(alt_path1):
         path_to_input_file = alt_path1
         found_file = True
         print(f"Info: Found city file at: {path_to_input_file}")

     # 2. Relative to script, in 'city-files' subdir
     if not found_file:
         alt_path2 = os.path.join("city-files", input_file)
         if os.path.isfile(alt_path2):
             path_to_input_file = alt_path2
             found_file = True
             print(f"Info: Found city file at: {path_to_input_file}")

     # 3. In the same directory as the script
     if not found_file:
         alt_path3 = input_file
         if os.path.isfile(alt_path3):
             path_to_input_file = alt_path3
             found_file = True
             print(f"Info: Found city file at: {path_to_input_file}")

     if not found_file:
        print(f"*** error: The city file '{input_file}' could not be found in standard locations.")
        sys.exit()

# Proceed with reading the file now that path_to_input_file is confirmed
ord_range = [[32, 126]]
file_string = read_file_into_string(path_to_input_file, ord_range)
file_string = remove_all_spaces(file_string)
print("I have found and read the input file " + input_file + ":")


location = file_string.find("SIZE=")
if location == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted (missing SIZE=).")
    sys.exit()

comma = file_string.find(",", location)
if comma == -1:
    print("*** error: The city file " + input_file + " is incorrectly formatted (missing comma after SIZE=).")
    sys.exit()

num_cities_as_string = file_string[location + 5:comma]
num_cities = integerize(num_cities_as_string)
# Add check for valid number of cities
if num_cities <= 0:
    print(f"*** error: Invalid number of cities parsed: {num_cities}. Check SIZE= value in {input_file}.")
    sys.exit()
print("   the number of cities is stored in 'num_cities' and is " + str(num_cities))

# Adjust marker for distances based on the presence of "NOTE="
note_marker = "NOTE="
note_location = file_string.find(note_marker)
if note_location != -1:
    # Ensure the comma found is *before* the note, otherwise the format is odd
    if comma >= note_location:
         print(f"*** error: Found '{note_marker}' before the comma separating SIZE and distances in {input_file}.")
         sys.exit()
    string_for_distances = file_string[comma + 1:note_location]
else:
    # If NOTE= isn't found, assume distances go to the end of the string
    string_for_distances = file_string[comma + 1:]


distances = convert_to_list_of_int(string_for_distances)


# Adjust expected counts based on num_cities
full_count = num_cities * num_cities
upper_tri_count = (num_cities * (num_cities + 1)) // 2
strict_upper_tri_count = (num_cities * (num_cities - 1)) // 2

counted_distances = len(distances)

# Determine city_format based on counted_distances and num_cities
if counted_distances == full_count:
    city_format = "full"
elif counted_distances == upper_tri_count:
    city_format = "upper_tri"
elif counted_distances == strict_upper_tri_count:
    city_format = "strict_upper_tri"
else:
    print(f"*** error: The number of distances ({counted_distances}) does not match expected formats for {num_cities} cities (Full: {full_count}, UpperTri: {upper_tri_count}, StrictUpperTri: {strict_upper_tri_count}).")
    print(f"   distances parsed: {distances[:20]}...") # Print sample
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
# Path assumption for alg_codes: same as for city files (sibling to username folder)
path_for_alg_codes_and_tariffs = os.path.join("..", "alg_codes_and_tariffs.txt")
# Verify path exists
if not os.path.exists(path_for_alg_codes_and_tariffs):
     # Try path relative to script's current directory
     print(f"Warning: '{path_for_alg_codes_and_tariffs}' not found relative to script's parent directory.")
     path_for_alg_codes_and_tariffs = "alg_codes_and_tariffs.txt" # Assume same directory as script
     if not os.path.exists(path_for_alg_codes_and_tariffs):
         print(f"*** error: The file '{path_for_alg_codes_and_tariffs}' could not be found.")
         # Set flag to indicate file not found, but let the next block handle the exit
         flag = "not_exist"
         code_dictionary, tariff_dictionary = {}, {} # Ensure variables exist
     # else: print(f"Info: Found codes file at: {path_for_alg_codes_and_tariffs}") # Optional confirmation

############ END OF SECTOR 4 (IGNORE THIS COMMENT)

############ START OF SECTOR 5 (IGNORE THIS COMMENT)
# Check if the file was found in the previous block before trying to read
if 'flag' in locals() and flag == "not_exist":
     print("*** error: The text file 'alg_codes_and_tariffs.txt' does not exist.")
     sys.exit()
else:
     code_dictionary, tariff_dictionary, flag = read_in_algorithm_codes_and_tariffs(path_for_alg_codes_and_tariffs)
     if flag == "not_exist": # Should be caught above, but double-check
          print("*** error: The text file 'alg_codes_and_tariffs.txt' does not exist.")
          sys.exit()
     elif flag != "good":
         print(f"*** warning: Issues detected while reading 'alg_codes_and_tariffs.txt' (flag: {flag}). Check file format.")
         # Allow execution to continue but warn the user

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

my_user_name = "fnlz75" # <--- *** REPLACE WITH YOUR ACTUAL USERNAME ***

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

my_first_name = "YourFirstName" # <--- *** Optional: Replace ***
my_last_name = "YourLastName"   # <--- *** Optional: Replace ***

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

# Check if code_dictionary was successfully populated before accessing it
if not code_dictionary:
     print(f"*** warning: Algorithm code dictionary is empty. Cannot verify code '{algorithm_code}'.")
     # Optionally force to "XX" or proceed with the user-provided code
     # algorithm_code = "XX"
elif not algorithm_code in code_dictionary:
    print(f"*** warning: the algorithm code {algorithm_code} is not in 'alg_codes_and_tariffs.txt'. Using '{algorithm_code}' but description might be missing.")
    # algorithm_code = "XX" # Optional: Force to XX if not found
else:
    print("   your algorithm code is legal and is " + algorithm_code + " -" + code_dictionary.get(algorithm_code, "Description not found") + ".")


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

added_note = "Implementation of a PSO-MMAS hybrid algorithm with 2-opt local search for TSP." # Updated note

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

# --- PSO-MMAS Hybrid Implementation ---

# --- Helper Functions ---

def calculate_tour_length(tour, dist_matrix):
    """Calculates the total length of a given tour."""
    length = 0
    num_cities_in_tour = len(tour)
    if num_cities_in_tour == 0:
        return 0 # Handle empty tour case
    for i in range(num_cities_in_tour):
        u = tour[i]
        # Check if indices are within bounds of dist_matrix
        if not (0 <= u < len(dist_matrix)):
            print(f"*** Error: Invalid city index '{u}' in tour during length calculation.")
            return float('inf') # Return infinity or handle error appropriately
        # Ensure wrap-around index is also valid
        next_i = (i + 1) % num_cities_in_tour
        v = tour[next_i]
        if not (0 <= v < len(dist_matrix)):
            print(f"*** Error: Invalid city index '{v}' in tour during length calculation.")
            return float('inf')
        # Ensure sub-index is valid
        if not (0 <= v < len(dist_matrix[u])):
             print(f"*** Error: Invalid distance matrix access dist_matrix[{u}][{v}].")
             return float('inf')

        length += dist_matrix[u][v]
    return length


def generate_random_tour(num_cities):
    """Generates a random permutation of cities."""
    if num_cities <= 0: return [] # Handle invalid input
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tour

def get_swap_sequence(current_tour, target_tour):
    """
    Generates a sequence of swaps to transform current_tour towards target_tour.
    Simplified approach: find elements out of place and swap them.
    """
    if not current_tour or not target_tour or len(current_tour) != len(target_tour):
         print("Warning: Invalid input tours for get_swap_sequence.")
         return []

    swaps = []
    temp_tour = list(current_tour) # Work on a copy
    num_elements = len(temp_tour)
    positions = {city: i for i, city in enumerate(temp_tour)}

    for i in range(num_elements):
        # Check if the city at index i is already correct
        if temp_tour[i] != target_tour[i]:
            # City currently at index i
            current_city_at_i = temp_tour[i]
            # City that *should* be at index i according to the target tour
            target_city_for_i = target_tour[i]

            # Find where the target city currently is in our temp_tour
            try:
                target_city_current_pos = positions[target_city_for_i]
            except KeyError:
                # This shouldn't happen if both tours contain the same set of cities
                print(f"Error: City {target_city_for_i} from target tour not found in current tour's positions.")
                continue # Skip this step

            # Perform the swap in temp_tour
            temp_tour[i], temp_tour[target_city_current_pos] = temp_tour[target_city_current_pos], temp_tour[i]

            # Record the swap by indices (i, target_city_current_pos)
            swaps.append((i, target_city_current_pos))

            # Update the positions dictionary for the two swapped cities
            positions[current_city_at_i] = target_city_current_pos
            positions[target_city_for_i] = i

    return swaps


def apply_swap_sequence(tour, swaps):
    """Applies a sequence of swaps to a tour."""
    new_tour = list(tour)
    tour_len = len(new_tour)
    for i, j in swaps:
         # Ensure indices are valid before swapping
         if 0 <= i < tour_len and 0 <= j < tour_len:
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
         else:
             # This indicates an issue, possibly with swap generation or tour modification
             print(f"Warning: Invalid swap indices ({i}, {j}) for tour length {tour_len}. Skipping swap.")
    return new_tour

# --- PSO-MMAS Parameters ---
max_it = 500         # Max iterations for the main loop
num_parts = 100      # Number of particles (PSO population size)
num_ants = 20        # Number of ants per ACO iteration

# PSO Parameters
w = 0.7              # Inertia weight
c1 = 1.5             # Cognitive coefficient
c2 = 1.5             # Social coefficient

# MMAS Parameters
alpha = 1.0          # Pheromone influence factor
beta = 3.0           # Heuristic influence factor
rho = 0.1            # Pheromone evaporation rate (MMAS often uses lower rho)
Q = 1.0              # Pheromone deposit base value (often 1 in MMAS, scaling comes from 1/Lgb)
mmas_a = 2 * num_cities # Factor for calculating tau_min = tau_max / mmas_a (adjust as needed)
epsilon = 1e-9       # Small value to prevent division by zero or log(0) issues

def calculate_tour_length(tour, dist_matrix):
    """Calculates the total length of a tour."""
    length = 0
    num_cities = len(tour)
    for i in range(num_cities):
        u = tour[i]
        v = tour[(i + 1) % num_cities]
        length += dist_matrix[u][v]
    return length

def apply_2opt(tour, dist_matrix):
    """Applies 2-opt local search to improve a tour."""
    n = len(tour)
    best_tour = tour[:]
    best_distance = calculate_tour_length(best_tour, dist_matrix)
    
    # For performance, limit iterations and check randomly
    max_iterations = 3  # Limit iterations to prevent slow execution
    
    for iteration in range(max_iterations):
        improved = False
        # Sample a limited set of random positions to check
        num_checks = min(20, n // 2)  # Limit the number of positions to check
        
        # Create a list of random i, j pairs to check
        i_range = range(1, n - 1)
        i_sample_size = min(num_checks, len(i_range))
        i_positions = random.sample(list(i_range), i_sample_size) if i_sample_size > 0 else []
        
        for i in i_positions:
            j_range = range(i + 1, n)
            j_sample_size = min(num_checks, len(j_range))
            j_positions = random.sample(list(j_range), j_sample_size) if j_sample_size > 0 else []
            
            for j in j_positions:
                # Calculate change if segment i to j is reversed
                current_edge1 = dist_matrix[best_tour[i-1]][best_tour[i]]
                current_edge2 = dist_matrix[best_tour[j]][best_tour[(j + 1) % n]]
                new_edge1 = dist_matrix[best_tour[i-1]][best_tour[j]]
                new_edge2 = dist_matrix[best_tour[i]][best_tour[(j + 1) % n]]

                delta = (new_edge1 + new_edge2) - (current_edge1 + current_edge2)

                if delta < -epsilon: # Use epsilon for floating point comparison
                    # Reverse the segment
                    best_tour[i:j+1] = best_tour[i:j+1][::-1]
                    best_distance += delta
                    improved = True
                    break  # Try another i
            
            if improved:
                break  # Move to next iteration
                
        if not improved:
            break  # If no improvement found, stop
            
    return best_tour, best_distance

# --- Initialization ---

# PSO Initialization
particles_pos = [generate_random_tour(num_cities) for _ in range(num_parts)]
particles_vel = [[] for _ in range(num_parts)] # Initial velocities are empty swap sequences
particles_pbest_pos = list(particles_pos)
particles_pbest_fit = [calculate_tour_length(p, dist_matrix) for p in particles_pbest_pos]

# Check for valid fitness values before finding min
valid_fits = [f for f in particles_pbest_fit if f != float('inf')]
if not valid_fits:
     print("*** Error: No valid initial tours found. Check distance matrix or tour generation.")
     # Handle error: maybe generate more tours, use a default high fitness, or exit
     global_best_fit = float('inf')
     global_best_pos = [] # Or a default invalid tour
     # sys.exit() # Option to terminate
else:
    global_best_fit = min(valid_fits)
    global_best_pos = particles_pbest_pos[particles_pbest_fit.index(global_best_fit)]


# MMAS Initialization
heuristic = [[0.0 for _ in range(num_cities)] for _ in range(num_cities)]
for i in range(num_cities):
    for j in range(num_cities):
        if i != j:
             distance = dist_matrix[i][j]
             if distance > 0:
                heuristic[i][j] = 1.0 / distance
             elif distance == 0:
                  # Handle zero distance: assign a very high heuristic value
                  heuristic[i][j] = 1.0 / epsilon
             # else: Implicitly 0 if distance is negative or i==j


# Calculate initial tau_max and tau_min based on initial global best
if global_best_fit > 0 and global_best_fit != float('inf'):
    tau_max = 1.0 / (rho * global_best_fit)
else:
    # Handle case where initial best is 0 or infinity (e.g., only one city or error)
    # Use an estimated initial max pheromone based on average distance or a heuristic tour
    print("Warning: Initial global best fit is invalid for tau_max calculation. Estimating.")
    if num_cities > 1:
        avg_dist = sum(sum(row) for row in dist_matrix) / (num_cities * (num_cities -1) if num_cities > 1 else 1)
        estimated_len = avg_dist * num_cities if avg_dist > 0 else 1.0 # Avoid estimated_len=0
        tau_max = 1.0 / (rho * max(estimated_len, epsilon))
    else:
        tau_max = 1.0 # Default for single city case


tau_min = tau_max / mmas_a if mmas_a > 0 else epsilon # Ensure tau_min is positive

# Initialize pheromone matrix to tau_max
pheromone = [[tau_max for _ in range(num_cities)] for _ in range(num_cities)]
last_update_gbf = global_best_fit # Track gbf for recalculating limits


# --- Main Loop ---
print(f"\nStarting PSO-MMAS Hybrid Optimization for {max_it} iterations...")
for iteration in range(max_it):

    # --- PSO Phase ---
    for i in range(num_parts):
        # PSO Velocity and Position Update (same as before)
        current_pos = particles_pos[i]
        current_vel = particles_vel[i]
        pbest_pos = particles_pbest_pos[i]

        swaps_to_pbest = get_swap_sequence(current_pos, pbest_pos)
        swaps_to_gbest = get_swap_sequence(current_pos, global_best_pos)

        new_vel = []
        num_inertial_swaps = int(w * len(current_vel))
        if len(current_vel) > 0:
            new_vel.extend(random.sample(current_vel, min(num_inertial_swaps, len(current_vel))))

        num_pbest_swaps = int(c1 * random.random() * len(swaps_to_pbest))
        if len(swaps_to_pbest) > 0:
            new_vel.extend(random.sample(swaps_to_pbest, min(num_pbest_swaps, len(swaps_to_pbest))))

        num_gbest_swaps = int(c2 * random.random() * len(swaps_to_gbest))
        if len(swaps_to_gbest) > 0:
            new_vel.extend(random.sample(swaps_to_gbest, min(num_gbest_swaps, len(swaps_to_gbest))))

        new_vel = list(set(new_vel))
        particles_vel[i] = new_vel

        new_pos = apply_swap_sequence(current_pos, new_vel)
        particles_pos[i] = new_pos

        # Evaluate new position
        new_fit = calculate_tour_length(new_pos, dist_matrix)
        
        # Only apply 2-opt occasionally to save time (every 10th particle or if already better than average)
        if i % 10 == 0 or new_fit < sum(particles_pbest_fit)/len(particles_pbest_fit):
            improved_pos, improved_fit = apply_2opt(new_pos, dist_matrix)
            # Use the improved solution
            particles_pos[i] = improved_pos
            new_fit = improved_fit

        # Update personal best
        if new_fit < particles_pbest_fit[i]:
            particles_pbest_pos[i] = improved_pos
            particles_pbest_fit[i] = new_fit

            # Update global best
            if new_fit < global_best_fit:
                global_best_pos = improved_pos
                global_best_fit = new_fit
                print(f"  Iteration {iteration+1} (PSO): New global best found = {global_best_fit}")
                # MMAS specific: Need to potentially update tau limits here or flag for update

    # --- MMAS Phase ---

    # Check if global best has improved and recalculate tau limits if needed
    if global_best_fit < last_update_gbf and global_best_fit > 0:
        tau_max = 1.0 / (rho * global_best_fit)
        tau_min = tau_max / mmas_a
        last_update_gbf = global_best_fit
        # Optional: Reinitialize pheromones to new tau_max if strategy requires
        # pheromone = [[tau_max for _ in range(num_cities)] for _ in range(num_cities)]
        print(f"  Iteration {iteration+1}: Updated MMAS limits: tau_max={tau_max:.4g}, tau_min={tau_min:.4g}")


    # Pheromone Evaporation (on all trails)
    for r in range(num_cities):
        for s in range(num_cities):
            pheromone[r][s] *= (1.0 - rho)
            # Apply lower bound directly after evaporation (common MMAS variant)
            # pheromone[r][s] = max(pheromone[r][s], tau_min)


    # Ant Tour Construction (same as before)
    ant_tours = []
    ant_fits = []
    for ant in range(num_ants):
        current_city = random.randint(0, num_cities - 1)
        tour = [current_city]
        visited = {current_city}
        allowed_cities = set(range(num_cities)) - visited

        while allowed_cities:
            # Calculate probabilities based on current pheromone and heuristic
            prob_factors = []
            prob_cities = []
            total_prob_factor = 0.0

            for next_city in allowed_cities:
                 tau_val = max(pheromone[current_city][next_city], epsilon) # Use max with epsilon
                 eta_val = heuristic[current_city][next_city] # Assumes heuristic > 0 where possible
                 if eta_val > 0: # Only consider cities with valid heuristic
                     prob_factor = (tau_val ** alpha) * (eta_val ** beta)
                     prob_factors.append(prob_factor)
                     prob_cities.append(next_city)
                     total_prob_factor += prob_factor

            # Select next city
            if not prob_cities or total_prob_factor <= 0:
                # If no valid moves (disconnected graph or all factors zero)
                # Choose randomly from remaining allowed cities
                if allowed_cities:
                    next_city = random.choice(list(allowed_cities))
                else:
                    break # Should not happen if num_cities > 1 and connected graph
            else:
                probabilities = [factor / total_prob_factor for factor in prob_factors]
                chosen_list = random.choices(prob_cities, weights=probabilities, k=1)
                next_city = chosen_list[0]

            # Move ant
            tour.append(next_city)
            visited.add(next_city)
            allowed_cities.remove(next_city)
            current_city = next_city

        # Store ant's tour and fitness, update global best if needed
        if len(tour) == num_cities:
            # Calculate tour fitness
            fit = calculate_tour_length(tour, dist_matrix)
            
            # Only apply 2-opt to the most promising solutions
            if fit < global_best_fit * 1.5:  # Only if within 50% of the global best
                improved_tour, improved_fit = apply_2opt(tour, dist_matrix)
                ant_tours.append(improved_tour)
                ant_fits.append(improved_fit)
            else:
                ant_tours.append(tour)
                ant_fits.append(fit)
            
            # Check if we found a better solution
            current_fit = improved_fit if fit < global_best_fit * 1.5 else fit
            current_tour = improved_tour if fit < global_best_fit * 1.5 else tour
            
            if current_fit < global_best_fit:
                global_best_pos = current_tour
                global_best_fit = current_fit
                print(f"  Iteration {iteration+1} (ACO): New global best found = {global_best_fit}")
                # Flag to update tau limits in the next iteration or update now
                # Updating now might be slightly more reactive:
                if global_best_fit > 0: # Avoid division by zero
                    tau_max = 1.0 / (rho * global_best_fit)
                    tau_min = tau_max / mmas_a
                    last_update_gbf = global_best_fit
                    print(f"  Iteration {iteration+1}: Updated MMAS limits (intra-ACO): tau_max={tau_max:.4g}, tau_min={tau_min:.4g}")



    # MMAS Pheromone Deposit (only using global best tour)
    if global_best_fit > 0 and global_best_fit != float('inf'):
        delta_tau_gbest = Q / global_best_fit # Q=1 in typical MMAS, delta = 1/Lgb
        for k in range(num_cities):
            city1 = global_best_pos[k]
            city2 = global_best_pos[(k + 1) % num_cities]

            # Check indices are valid before updating
            if 0 <= city1 < num_cities and 0 <= city2 < num_cities:
                pheromone[city1][city2] += delta_tau_gbest
                pheromone[city2][city1] += delta_tau_gbest # Symmetric TSP
            else:
                 print(f"Warning: Invalid indices ({city1}, {city2}) in global best tour during pheromone deposit.")

    # Enforce Pheromone Limits (Min-Max) on all trails AFTER deposit
    for r in range(num_cities):
        for s in range(num_cities):
            pheromone[r][s] = max(tau_min, min(tau_max, pheromone[r][s]))


    # Optional: Print progress
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration+1}/{max_it} completed. Current Best Length: {global_best_fit}. Tau Limits: [{tau_min:.3g}, {tau_max:.3g}]")


# --- Final Result ---
# Apply one final thorough 2-opt optimization to the best tour found
# Define a more aggressive 2-opt function just for the final optimization
def thorough_2opt(tour, dist_matrix, max_iter=10):
    """More thorough 2-opt search for the final optimization"""
    n = len(tour)
    best_tour = tour[:]
    best_distance = calculate_tour_length(best_tour, dist_matrix)
    
    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate change if segment i to j is reversed
                current_edge1 = dist_matrix[best_tour[i-1]][best_tour[i]]
                current_edge2 = dist_matrix[best_tour[j]][best_tour[(j + 1) % n]]
                new_edge1 = dist_matrix[best_tour[i-1]][best_tour[j]]
                new_edge2 = dist_matrix[best_tour[i]][best_tour[(j + 1) % n]]

                delta = (new_edge1 + new_edge2) - (current_edge1 + current_edge2)

                if delta < -epsilon:
                    # Reverse the segment
                    best_tour[i:j+1] = best_tour[i:j+1][::-1]
                    best_distance += delta
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_tour, best_distance

print("Performing final optimization...")
final_tour, final_tour_length = thorough_2opt(global_best_pos, dist_matrix)
tour = final_tour
tour_length = int(round(final_tour_length)) if final_tour_length != float('inf') else -1 # Handle potential error case

# Add parameters to added_note
added_note += f"\nPSO Params: w={w}, c1={c1}, c2={c2}, num_parts={num_parts}."
added_note += f"\nMMAS Params: alpha={alpha}, beta={beta}, rho={rho}, Q={Q}, mmas_a={mmas_a}, num_ants={num_ants}."
added_note += f"\nMax Iterations: {max_it}."


print(f"\nOptimization finished. Best tour length found: {tour_length} (original: {int(round(global_best_fit))}, 2-opt improved: {tour_length})")


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

# --- Adjust note generation for hybrid algorithm ---
# We have max_it, num_parts, and num_ants defined above

if added_note != "":
    added_note = added_note + "\n"
# Combine parameters into the note regardless of algorithm_code specifics in the original template
added_note = added_note + f"Parameters: max_it={max_it}, num_parts={num_parts}, num_ants={num_ants}."


added_note = added_note + "\nRUN-TIME = " + str(elapsed_time) + " seconds.\n"
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
added_note = added_note + "DATE-TIME = " + dt_string + ".\n"


# --- Validation and Output Code (Original Skeleton) ---
flag = "good"

# Check if tour is valid before proceeding
if not isinstance(tour, list) or not tour:
     print("*** error: Final 'tour' is not a valid list or is empty.")
     sys.exit()
if tour_length == -1: # Check if optimization failed to find a valid tour length
     print("*** error: Final 'tour_length' indicates an issue during optimization (-1).")
     sys.exit()


length = len(tour)
for i in range(0, length):
    if isinstance(tour[i], int) == False:
        flag = "bad"
    else:
        tour[i] = int(tour[i]) # Ensure integer type
if flag == "bad":
    print("*** error: Your final tour contains non-integer values.")
    sys.exit()

# Check if tour_length exists and is calculable, otherwise calculate it
try:
    tour_length
    if not isinstance(tour_length, (int, float)):
        print("*** warning: 'tour_length' is not a number. Recalculating.")
        tour_length = calculate_tour_length(tour, dist_matrix)
except NameError:
    print("*** warning: 'tour_length' not explicitly assigned. Calculating.")
    tour_length = calculate_tour_length(tour, dist_matrix)

# Ensure tour_length is an integer for the final check
tour_length = int(round(tour_length))


if len(tour) != num_cities:
    print("*** error: The final tour does not consist of " + str(num_cities) + " cities as there are, in fact, " + str(len(tour)) + ".")
    sys.exit()

# Check for uniqueness and range more robustly
city_set = set(tour)
if len(city_set) != num_cities:
    print("*** error: Your final tour has repeated or missing city names.")
    # Find duplicates/missing for better debugging
    counts = {}
    for city in tour: counts[city] = counts.get(city, 0) + 1
    duplicates = [c for c, count in counts.items() if count > 1]
    missing = [c for c in range(num_cities) if c not in city_set]
    if duplicates: print(f"   Duplicates found: {duplicates}")
    if missing: print(f"   Missing cities: {missing}")
    sys.exit()

# Check if all cities 0..N-1 are present
expected_cities = set(range(num_cities))
if city_set != expected_cities:
     print(f"*** error: Final tour cities {city_set} do not match expected {expected_cities}.")
     missing = list(expected_cities - city_set)
     extra = list(city_set - expected_cities)
     if missing: print(f"   Missing cities: {missing}")
     if extra: print(f"   Unexpected cities: {extra}")
     sys.exit()


# Recalculate tour length for final verification
check_tour_length = 0
try:
    check_tour_length = calculate_tour_length(tour, dist_matrix)
    if check_tour_length == float('inf'):
        print("*** error: Recalculation of final tour length failed (returned infinity). Check tour validity.")
        sys.exit()
except Exception as e:
     print(f"*** error: Exception during final tour length recalculation: {e}")
     sys.exit()


# Compare stated vs calculated length, allowing for minor float inaccuracies before rounding
tolerance = 1e-6 # Tolerance for floating point comparison
if abs(float(tour_length) - check_tour_length) > tolerance:
    print(f"*** warning: The final stated tour length {tour_length} differs significantly from the recalculated length {check_tour_length:.4f}. Using recalculated value for output.")
    tour_length = int(round(check_tour_length)) # Use the verified length


print("You, user " + my_user_name + ", have successfully built a tour of length " + str(tour_length) + "!")
len_user_name = len(my_user_name)
user_number = 0
for i in range(0, len_user_name):
    user_number = user_number + ord(my_user_name[i])

alg_number = 0
# Ensure algorithm_code is a string with at least 2 chars before calculating alg_number
if isinstance(algorithm_code, str) and len(algorithm_code) >= 2 :
   try:
       alg_number = ord(algorithm_code[0]) + ord(algorithm_code[1])
   except TypeError:
        print(f"Warning: Could not calculate algorithm number from code '{algorithm_code}'. Using 0.")
        alg_number = 0
else:
     print(f"Warning: Algorithm code '{algorithm_code}' is invalid for number calculation. Using 0.")
     alg_number = 0


len_dt_string = len(dt_string)
date_time_number = 0
for i in range(0, len_dt_string):
    date_time_number = date_time_number + ord(dt_string[i])

tour_diff = 0
if num_cities > 0 and tour: # Check if tour is not empty
    try:
        tour_diff = abs(tour[0] - tour[num_cities - 1])
        for i in range(0, num_cities - 1):
            tour_diff = tour_diff + abs(tour[i + 1] - tour[i])
    except IndexError:
         print("*** error: Index out of bounds calculating tour difference. Check tour structure.")
         # Handle error, maybe set tour_diff to 0 or exit
         tour_diff = 0
         # sys.exit()


certificate = user_number + alg_number + date_time_number + tour_diff
local_time = time.asctime(time.localtime(time.time()))
output_file_time = local_time[4:7] + local_time[8:10] + local_time[11:13] + local_time[14:16] + local_time[17:19]
output_file_time = output_file_time.replace(" ", "0")
script_name = os.path.basename(sys.argv[0])
if len(sys.argv) > 2:
    output_file_time = sys.argv[2]

# Ensure script_name and input_file parts are valid before creating the filename
script_base = script_name.rsplit('.', 1)[0] if '.' in script_name else script_name
input_base = input_file.rsplit('.', 1)[0] if '.' in input_file else input_file

# Use the potentially modified algorithm_code for the filename
output_file_name = f"{script_base}_{input_base}_{algorithm_code}_{output_file_time}.txt"


try:
    f = open(output_file_name,'w')
    f.write("USER = {0} ({1} {2}),\n".format(my_user_name, my_first_name, my_last_name))
    f.write("ALGORITHM CODE = {0}, NAME OF CITY-FILE = {1},\n".format(algorithm_code, input_file))
    f.write("SIZE = {0}, TOUR LENGTH = {1},\n".format(num_cities, tour_length))
    if tour: # Check if tour is not empty before writing
        f.write(str(tour[0]))
        for i in range(1,num_cities):
            f.write(",{0}".format(tour[i]))
    f.write(",\nNOTE = {0}".format(added_note))
    f.write("CERTIFICATE = {0}.\n".format(certificate))
    f.close()
    print("I have successfully written your tour to the tour file:\n   " + output_file_name + ".")
except IOError as e:
    print(f"*** error: Could not write tour file '{output_file_name}'. Reason: {e}")
except Exception as e:
     print(f"*** error: An unexpected error occurred during file writing: {e}")


############ END OF SECTOR 10 (IGNORE THIS COMMENT)