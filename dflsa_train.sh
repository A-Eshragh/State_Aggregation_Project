#!/bin/bash

#SBATCH -o log/train_%a.log
#SBATCH -p parallel
#SBATCH -N 1
#SBATCH --mem=28G
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-11
#SBATCH -t 24:00:00

# Create the log directory if not exists
OUTPUT_DIR="log"
mkdir -p $OUTPUT_DIR

python dflsa_train.py "$@"
