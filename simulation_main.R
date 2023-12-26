###########################################################
# Purpose: simulation
# Date: 2023-12-06
# Author: Chung-Kang Lo
###########################################################

rm(list = ls())

#!/usr/bin/env Rscript

# Access the arguments
args <- commandArgs(trailingOnly = TRUE)

# Set seed for reproducibility
set.seed(1234)

# YOUR DIRECTORY
dir <- args[1]
output_dir <- paste0(args[1], "/", args[2])

# Import functions
source(paste0(dir, '/uniform_policy.R'))
source(paste0(dir, '/first_best_policy.R'))
source(paste0(dir, '/policy_learning.R'))

# Generate samples
# generate_samples(dir)

outcome <- c("outcome")
setting <- c(1, 2, 3, 4, 5)
cost <- c(0, 0.5, 0.7, 0.9)

# # Uniform policy
for (iii in 1:10){
  uniform_policy(output_dir, outcome, cost, iii-1)
}

# First Best Policy
for (iii in 1:10){
  first_best_policy(output_dir, outcome, setting, cost, iii-1)
}

# Policy Learning
for (iii in 1:10){
  policy_learning(output_dir, outcome, setting, cost, iii-1)
}
