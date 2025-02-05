#!/bin/bash

#SBATCH -o log/task_%a.log
#SBATCH -p parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-47
#SBATCH -t 24:00:00

# Create the log directory if not exists
OUTPUT_DIR="log"
mkdir -p $OUTPUT_DIR

python dflsa_train.py
