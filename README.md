# AI Search Algorithms for the Traveling Salesman Problem

A Python implementation of metaheuristic algorithms to solve the Traveling Salesman Problem (TSP), developed as a computer science coursework project.

## Project Overview

This project implements and compares two bio-inspired optimization algorithms for solving TSP:
- **Ant Colony Optimization (ACO)** - Algorithm A
- **Particle Swarm Optimization (PSO)** - Algorithm B

Both algorithms are implemented in basic and enhanced versions to demonstrate optimization techniques and parameter tuning.

## Author

**Morgan Rosca** (fnlz75)

## Algorithms Implemented

### Algorithm A: Ant Colony Optimization (ACO)
- **Algorithm Code**: AC
- **Tariff**: 9
- **Implementation**: Elitist Ant System (EAS) with pheromone-based path construction
- **Key Features**:
  - Dynamic ant population sizing based on problem size
  - Pheromone evaporation and deposit mechanisms
  - Visibility heuristic (inverse distance)
  - Elitist reinforcement for best-so-far tours
  - Probabilistic city selection using pheromone trails

**Parameters**:
- `alpha` = 1.0 (pheromone importance)
- `beta` = 5.0 (heuristic importance)
- `rho` = 0.5 (evaporation rate)
- `Q` = 100 (pheromone deposit factor)
- `elitist_weight` = 5

### Algorithm B: Particle Swarm Optimization (PSO)
- **Algorithm Code**: PS
- **Tariff**: 10
- **Implementation**: Discrete PSO with swap-based velocity operators
- **Key Features**:
  - Tour representation as particle positions
  - Velocity represented as swap sequences
  - Personal best (pbest) and global best (gbest) tracking
  - Inertia, cognitive, and social components
  - Random tour initialization with shuffle

**Parameters**:
- `num_parts` = 60 (swarm size)
- `max_iter` = 1000 (maximum iterations)
- `w_start` = 0.5 (inertia weight)
- `c1` = 1.0 (cognitive weight)
- `c2` = 1.0 (social weight)

## Project Structure

```
AISearch/
├── README.md                           # This file
├── skeleton.py                         # Template program provided by instructor
├── alg_codes_and_tariffs.txt          # Algorithm codes and difficulty tariffs
├── validate_before_handin.py          # Validation script for submissions
├── city-files/                        # Test data directory
│   ├── AISearchfile012.txt           # 12 cities
│   ├── AISearchfile017.txt           # 17 cities
│   ├── AISearchfile021.txt           # 21 cities
│   ├── AISearchfile026.txt           # 26 cities
│   ├── AISearchfile042.txt           # 42 cities
│   ├── AISearchfile048.txt           # 48 cities
│   ├── AISearchfile058.txt           # 58 cities
│   ├── AISearchfile175.txt           # 175 cities
│   ├── AISearchfile180.txt           # 180 cities
│   └── AISearchfile535.txt           # 535 cities
└── fnlz75/                            # Implementation directory
    ├── AlgAbasic.py                  # Basic ACO implementation
    ├── AlgAenhanced.py               # Enhanced ACO implementation
    ├── AlgBbasic.py                  # Basic PSO implementation
    ├── AlgBenhanced.py               # Enhanced PSO implementation
    ├── AlgA_AISearchfile*.txt        # ACO tour results (10 files)
    ├── AlgB_AISearchfile*.txt        # PSO tour results (10 files)
    └── AISearchValidationFeedback.txt # Validation results
```

## Input File Format

City files contain distance matrices in the following format:

```
SIZE = n,
d(0,1), d(0,2), ..., d(0,n-1),
d(1,2), d(1,3), ..., d(1,n-1),
...
NOTE =
```

The distance matrix can be provided in three formats:
- **Full matrix**: n × n values
- **Upper triangular**: n(n+1)/2 values
- **Strict upper triangular**: n(n-1)/2 values

## Usage

### Running the Algorithms

```bash
# Run ACO on a specific city file
cd fnlz75
python AlgAbasic.py AISearchfile012.txt

# Run PSO on a specific city file
python AlgBbasic.py AISearchfile012.txt
```

### Running Validation

```bash
python validate_before_handin.py
```

This validates:
- Program files (user ID, algorithm codes, imports)
- Tour files (validity, tour length calculations)
- Required submission files

## Results

### Validation Status
All programs and tours successfully validated:
- ✅ Program cluster AlgA: Valid (ACO, tariff 9)
- ✅ Program cluster AlgB: Valid (PSO, tariff 10)
- ✅ All 20 tour files validated

### Performance Comparison

| City File | Size | ACO Length | ACO Time (s) | PSO Length | PSO Time (s) |
|-----------|------|------------|--------------|------------|--------------|
| AISearchfile012 | 12 | 56 | 0.6 | 56 | 1.5 |
| AISearchfile017 | 17 | 1,444 | 0.9 | 1,444 | 2.8 |
| AISearchfile021 | 21 | 2,549 | 1.1 | 2,549 | 4.1 |
| AISearchfile026 | 26 | 1,473 | 1.4 | 1,473 | 6.1 |
| AISearchfile042 | 42 | 1,190 | 2.8 | 1,196 | 16.6 |
| AISearchfile048 | 48 | 12,216 | 3.2 | 12,299 | 21.7 |
| AISearchfile058 | 58 | 25,395 | 4.5 | 25,845 | 13.7 |
| AISearchfile175 | 175 | 21,431 | 39.2 | 21,810 | 100.4 |
| AISearchfile180 | 180 | 1,950 | 44.3 | 1,950 | 93.7 |
| AISearchfile535 | 535 | 48,550 | 678.1 | 49,141 | 1,515.6 |

### Key Observations

1. **Solution Quality**: Both algorithms find optimal or near-optimal solutions for small instances (≤26 cities)
2. **ACO Advantages**:
   - Generally finds better solutions, especially on larger instances
   - Significantly faster runtime (2-3x faster than PSO on large instances)
3. **PSO Characteristics**:
   - Competitive on smaller instances
   - Runtime scales less favorably with problem size

## Algorithm Details

### Ant Colony Optimization (ACO)

The ACO implementation uses an Elitist Ant System approach:

1. **Tour Construction**: Each ant builds a tour probabilistically using:
   - Pheromone levels (τ): learned component
   - Visibility (η = 1/distance): greedy heuristic
   - Selection probability: proportional to τ^α × η^β

2. **Pheromone Update**:
   - Evaporation: τ(i,j) ← (1-ρ) × τ(i,j)
   - Deposit: All ants deposit pheromone inversely proportional to tour length
   - Elitist reinforcement: Extra pheromone on best-so-far tour

3. **Dynamic Parameters**:
   - Ant count scales with problem size
   - Iterations adjusted to maintain computational budget

### Particle Swarm Optimization (PSO)

The PSO implementation adapts the continuous PSO paradigm to discrete tours:

1. **Position Representation**: Tour as permutation of cities
2. **Velocity Representation**: Sequence of swap operations
3. **Velocity Update**:
   - Inertia: maintains current search direction
   - Cognitive: attracts particle to its personal best
   - Social: attracts particle to global best
4. **Position Update**: Apply swap sequence to current tour

## Technical Requirements

- **Language**: Python 3.x
- **Standard Libraries Only**: os, sys, time, random, datetime, math
- **No External Dependencies**: Pure Python implementation

## Implementation Notes

- Both algorithms use only Python standard library modules
- Tour validation ensures all cities visited exactly once
- Output files include algorithm parameters and execution metadata
- Certificate generation for submission verification

## Future Enhancements

Potential improvements for the enhanced versions:
- 2-opt local search for ACO tours
- MAX-MIN Ant System (MMAS) pheromone limits
- Adaptive parameter tuning for PSO
- Hybrid approaches combining both algorithms
- Parallel ant/particle evaluation

## License

Academic coursework project - All rights reserved

## Acknowledgments

Template code and validation framework provided by course instructor.
