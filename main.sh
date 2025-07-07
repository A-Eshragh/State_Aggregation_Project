#!/bin/bash

#SBATCH -p parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH -t 72:00:00

# Create the log directory if not exists
mkdir -p log

# ------------------------ run Python ------------------------------
# Forward *all* command line input args the user gave to sbatch, plus the derived --task_id.
# Example:
# sbatch -o log/train.log main.sh --mode train --hyperparam "{'lr': 0.01, 'hidden_dim': 32}"
#   →  This command runs main.py in training mode with the specified hyperparameters and saves to log/train.log
# sbatch -o log/train_%a.log --array=0-7 main.sh --mode train
#   →  This command runs main.py in training mode with hyperparameter tuning and saves each task to its own log file
# sbatch -o log/test.log main.sh --mode test
#   →  This command runs main.py in test mode with the best hyperparameter and saves to log/test.log
python main.py "$@"
