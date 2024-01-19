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
source(paste0(dir, '/statistics_graph.R'))

# parameters
num_simulation <- args[3]

# setting
outcome <- c("outcome")
setting <- c(1, 2, 3, 4, 5, 6, 7)
cost <- c(0, 0.5, 0.7, 0.9)

# Calcuate welfare for different strategy
# Uniform policy
for (iii in 1:num_simulation){
  uniform_policy(output_dir, outcome, cost, iii-1)
}

# First Best Policy
for (iii in 1:num_simulation){
  first_best_policy(output_dir, outcome, setting, cost, iii-1)
}

# Policy Learning
for (iii in 1:num_simulation){
  policy_learning(output_dir, outcome, setting, cost, iii-1)
}

# Treatment effect visualization
for (iii in 1:num_simulation){
  statistics_graph(output_dir, outcome, setting, cost, iii-1)
}
