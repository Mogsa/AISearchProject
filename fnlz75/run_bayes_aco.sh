#!/bin/bash
# This script runs BayesianACO.py with the correct environment setup

# Check if conda is available
if command -v conda &> /dev/null; then
    # Create a conda environment if it doesn't exist
    if ! conda env list | grep -q "aco_opt"; then
        echo "Creating conda environment for ACO optimization..."
        conda create -y -n aco_opt python=3.9
        conda activate aco_opt
        conda install -y numpy=1.23 scikit-optimize scikit-learn matplotlib pandas
    else
        conda activate aco_opt
    fi
else
    # If conda not available, try using pip in a virtual environment
    if [ ! -d "$HOME/venv-aco" ]; then
        echo "Creating virtual environment for ACO optimization..."
        python -m venv $HOME/venv-aco
        source $HOME/venv-aco/bin/activate
        pip install numpy==1.23.5 scikit-optimize scikit-learn matplotlib pandas
    else
        source $HOME/venv-aco/bin/activate
    fi
fi

# Run the script with any arguments passed to this script
python /Users/morgan/Documents/GitHub/AISearch/fnlz75/BayesianACO.py "$@"

# Deactivate the environment
if command -v conda &> /dev/null; then
    conda deactivate
else
    deactivate
fi