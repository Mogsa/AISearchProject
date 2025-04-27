import os
import sys
import time
import random
from datetime import datetime
import csv
import numpy as np

# Import Bayesian optimization tools
from skopt import gp_minimize, dump, load
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Import necessary functions from AlphaBetaSearch.py
# We'll reuse the ACO code and evaluation functions
from AlphaBetaSearch import (
    run_aco_with_params, 
    read_file_into_string, 
    remove_all_spaces, 
    integerize, 
    convert_to_list_of_int, 
    build_distance_matrix
)

class BayesianACOOptimizer:
    """1
    Bayesian optimization for ACO parameters using Gaussian Process regression.
    This approach is more efficient than grid search, especially for expensive 
    function evaluations like ACO runs on large TSP instances.
    """
    
    def __init__(self, city_file, max_evals=30, trials_per_eval=2, 
                 fixed_rho=0.15, fixed_cl_size=20):
        """
        Initialize the Bayesian optimizer for ACO parameters.
        
        Parameters:
        - city_file: Name of the city file to use for optimization
        - max_evals: Maximum number of evaluations to perform
        - trials_per_eval: Number of ACO trials per parameter combination (for robustness)
        - fixed_rho: Fixed evaporation rate
        - fixed_cl_size: Fixed candidate list size
        
        Note: lambda_vis (adaptive visibility weight) is now optimized along with alpha and beta
        """
        self.city_file = city_file
        self.max_evals = max_evals
        self.trials_per_eval = trials_per_eval
        
        # Fixed parameters
        self.fixed_rho = fixed_rho
        self.fixed_cl_size = fixed_cl_size
        
        # Define the parameter search space
        self.search_space = [
            Real(0.1, 5.0, name='alpha'),     # Pheromone importance
            Real(0.5, 10.0, name='beta'),     # Heuristic importance
            Real(0.0, 2.0, name='lambda_vis') # Adaptive visibility weight
        ]
        
        # Setup results tracking
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = f"bayesian_aco_{city_file.replace('.txt', '')}_{self.timestamp}.csv"
        self.evaluations = []
        
        # Initialize results file
        with open(self.results_file, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Evaluation", "Alpha", "Beta", "Lambda_vis", "Avg Length", "Best Length"])
        
        # Create the decorated objective function with correctly initialized dimensions
        self._raw_objective_function = self._create_objective_function()
        self.objective_function = use_named_args(dimensions=self.search_space)(self._raw_objective_function)
    
    def _create_objective_function(self):
        """
        Create the raw objective function to be decorated
        """
        def func(alpha, beta, lambda_vis):
            """
            Objective function for Bayesian optimization.
            Returns the average tour length (to be minimized).
            """
            print(f"\nEvaluating: alpha={alpha:.3f}, beta={beta:.3f}, lambda_vis={lambda_vis:.3f}")
            
            # Run ACO with the specified parameters
            avg_length, best_length = run_aco_with_params(
                alpha, beta, self.city_file, 
                self.fixed_rho, lambda_vis, self.fixed_cl_size, 
                num_trials=self.trials_per_eval
            )
            
            # Log the result
            eval_num = len(self.evaluations) + 1
            result = (eval_num, alpha, beta, lambda_vis, avg_length, best_length)
            self.evaluations.append(result)
            
            # Write to CSV
            with open(self.results_file, "a", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(result)
            
            print(f"Result: avg_length={avg_length:.2f}, best_length={best_length}")
            
            return avg_length  # Return the value to be minimized
        
        return func
    
    def optimize(self):
        """
        Run the Bayesian optimization process.
        Returns the optimal parameters and results.
        """
        print(f"Starting Bayesian optimization for {self.city_file}")
        print(f"Will perform up to {self.max_evals} evaluations, {self.trials_per_eval} trials each")
        
        # Run Bayesian optimization
        result = gp_minimize(
            self.objective_function,      # the function to minimize
            self.search_space,            # the search space
            n_calls=self.max_evals,       # number of evaluations
            n_initial_points=5,           # initial random evaluations
            random_state=123,             # for reproducibility
            verbose=True,                 # print progress
            n_jobs=1                      # serial processing 
        )
        
        # Get optimal parameters
        optimal_alpha = result.x[0]
        optimal_beta = result.x[1]
        optimal_lambda_vis = result.x[2]
        
        # Save optimization results in a simpler way - don't try to pickle the result object
        # Instead, save just the important data
        best_params = {
            'alpha': optimal_alpha,
            'beta': optimal_beta,
            'lambda_vis': optimal_lambda_vis,
            'score': result.fun,
            'params_tried': [(x[0], x[1], x[2]) for x in self.X] if hasattr(self, 'X') else [],
            'scores': self.y if hasattr(self, 'y') else [],
            'city_file': self.city_file,
            'timestamp': self.timestamp
        }
        
        # Save as a simple CSV instead of pickle
        result_file = f'bayesian_result_{self.city_file.replace(".txt", "")}_{self.timestamp}.csv'
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Alpha', optimal_alpha])
            writer.writerow(['Beta', optimal_beta])
            writer.writerow(['Lambda_vis', optimal_lambda_vis])
            writer.writerow(['Best Score', result.fun])
            writer.writerow(['City File', self.city_file])
        
        print("\nOptimization complete!")
        print(f"Best parameters: alpha={optimal_alpha:.3f}, beta={optimal_beta:.3f}, lambda_vis={optimal_lambda_vis:.3f}")
        print(f"Best score (avg length): {result.fun:.2f}")
        print(f"All evaluations saved to {self.results_file}")
        print(f"Results summary saved to {result_file}")
        
        return optimal_alpha, optimal_beta, optimal_lambda_vis, result.fun, self.evaluations

def batch_bayesian_optimization(city_files=None, trials_per_eval=2, evals_per_city=None):
    """
    Run Bayesian optimization for multiple city files in sequence.
    
    Parameters:
    - city_files: List of city files to process (or None for default list)
    - trials_per_eval: Number of ACO trials per evaluation
    - evals_per_city: Dictionary mapping city files to max evaluations
    """
    if city_files is None:
        # Default list of city files, ordered by size
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
    
    # Define max evaluations per city file (more for smaller problems, fewer for larger ones)
    if evals_per_city is None:
        evals_per_city = {
            "AISearchfile012.txt": 30,  # Small, can do more evaluations
            "AISearchfile017.txt": 30,
            "AISearchfile021.txt": 30,
            "AISearchfile026.txt": 25,
            "AISearchfile042.txt": 25,
            "AISearchfile048.txt": 20,
            "AISearchfile058.txt": 20,
            "AISearchfile175.txt": 15,  # Larger, reduce evaluations
            "AISearchfile180.txt": 15,
            "AISearchfile535.txt": 10   # Very large, minimum evaluations
        }
    
    # Create a summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"bayesian_summary_{timestamp}.csv"
    
    with open(summary_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["City File", "Optimal Alpha", "Optimal Beta", "Optimal Lambda_vis", "Best Avg Length", "Evaluations"])
        
        # Process each city file
        for city_file in city_files:
            print(f"\n{'='*80}")
            print(f"PROCESSING CITY FILE: {city_file}")
            print(f"{'='*80}")
            
            try:
                # Get max evaluations for this city
                max_evals = evals_per_city.get(city_file, 20)  # Default to 20 if not specified
                
                # Create and run optimizer
                optimizer = BayesianACOOptimizer(
                    city_file=city_file,
                    max_evals=max_evals,
                    trials_per_eval=trials_per_eval
                )
                
                optimal_alpha, optimal_beta, optimal_lambda_vis, best_score, _ = optimizer.optimize()
                
                # Write to summary
                csvwriter.writerow([city_file, optimal_alpha, optimal_beta, optimal_lambda_vis, best_score, max_evals])
                csvfile.flush()
            
            except Exception as e:
                print(f"Error processing {city_file}: {e}")
                csvwriter.writerow([city_file, "ERROR", str(e), "", ""])
                csvfile.flush()
    
    print(f"\nBatch optimization complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    print("Bayesian Optimization for ACO Parameters")
    print("========================================")
    print("This tool uses Bayesian optimization to efficiently find optimal ACO parameters.")
    print("\nChoose run mode:")
    print("1. Single city file")
    print("2. Batch process all city files")
    
    mode = input("Enter your choice (1-2): ")
    
    if mode == "2":
        # Batch processing mode
        print("\nBatch Processing Mode Selected")
        print("This will process all city files in sequence using Bayesian optimization.")
        
        # Get trials per evaluation
        try:
            trials_per_eval = int(input("Enter number of ACO trials per evaluation (1-5, default=2): "))
            if trials_per_eval < 1 or trials_per_eval > 5:
                print("Using default value of 2 trials")
                trials_per_eval = 2
        except:
            print("Using default value of 2 trials")
            trials_per_eval = 2
        
        # Allow custom max evaluations
        custom_evals = input("Do you want to customize the number of evaluations per city file? (y/n, default=n): ").lower() == 'y'
        evals_per_city = None
        
        if custom_evals:
            evals_per_city = {}
            print("\nEnter max evaluations for each city file (larger files should have fewer to save time):")
            city_files = [
                "AISearchfile012.txt", "AISearchfile017.txt", "AISearchfile021.txt",
                "AISearchfile026.txt", "AISearchfile042.txt", "AISearchfile048.txt",
                "AISearchfile058.txt", "AISearchfile175.txt", "AISearchfile180.txt",
                "AISearchfile535.txt"
            ]
            
            for city_file in city_files:
                default_evals = 30 if int(city_file[13:16]) < 100 else (15 if int(city_file[13:16]) < 200 else 10)
                try:
                    max_evals = int(input(f"{city_file} (default={default_evals}): ") or default_evals)
                    evals_per_city[city_file] = max_evals
                except:
                    evals_per_city[city_file] = default_evals
        
        # Confirm before starting
        confirm = input("\nThis will run for several hours optimizing all city files.\nDo you want to proceed? (y/n): ").lower()
        if confirm == 'y':
            batch_bayesian_optimization(trials_per_eval=trials_per_eval, evals_per_city=evals_per_city)
        else:
            print("Operation cancelled.")
    
    else:
        # Single city file mode
        if len(sys.argv) > 1:
            city_file = sys.argv[1]
        else:
            city_file = input("Enter city file name (default: AISearchfile180.txt): ") or "AISearchfile180.txt"
        
        print(f"\nBayesian optimization for ACO on city file: {city_file}")
        
        # Get optimization parameters
        try:
            max_evals = int(input("Enter maximum number of evaluations (10-50, default=30): ") or "30")
            if max_evals < 10 or max_evals > 50:
                print("Using default value of 30 evaluations")
                max_evals = 30
        except:
            print("Using default value of 30 evaluations")
            max_evals = 30
        
        try:
            trials_per_eval = int(input("Enter number of ACO trials per evaluation (1-5, default=3): ") or "3")
            if trials_per_eval < 1 or trials_per_eval > 5:
                print("Using default value of 3 trials")
                trials_per_eval = 3
        except:
            print("Using default value of 3 trials")
            trials_per_eval = 3
        
        # Run optimizer
        optimizer = BayesianACOOptimizer(
            city_file=city_file,
            max_evals=max_evals,
            trials_per_eval=trials_per_eval
        )
        
        optimal_alpha, optimal_beta, optimal_lambda_vis, best_score, _ = optimizer.optimize()
        
        print("\nOptimization Results:")
        print(f"Optimal Alpha: {optimal_alpha:.4f}")
        print(f"Optimal Beta: {optimal_beta:.4f}")
        print(f"Optimal Lambda_vis: {optimal_lambda_vis:.4f}")
        print(f"Best Average Tour Length: {best_score:.2f}")