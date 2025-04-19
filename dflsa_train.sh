#!/bin/bash

#SBATCH -o log/train_%a.log
#SBATCH -p parallel
#SBATCH -N 1
#SBATCH --mem=36G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-15
#SBATCH -t 48:00:00

# Create the log directory if not exists
OUTPUT_DIR="log"
mkdir -p $OUTPUT_DIR

python dflsa_train.py "$@"
