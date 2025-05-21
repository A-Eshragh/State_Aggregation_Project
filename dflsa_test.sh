#!/bin/bash

#SBATCH -o log/test.log
#SBATCH -p parallel
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00

# Create the log directory if not exists
OUTPUT_DIR="log"
mkdir -p $OUTPUT_DIR

python dflsa_test.py "$@"
