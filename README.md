# Simulation Task for Welfare Estimation README

## Overview
This README provides instructions for a comprehensive simulation task designed for welfare estimation. The task employs a novel targeting method that merges causal inference with the Poisson Hidden Markov Model (PHMM), offering a robust framework for understanding welfare dynamics.

## Getting Started
Before starting, ensure that you have the required software and dependencies installed. This simulation is typically run on platforms that support `.sh` and `.py` scripts, as well as R scripts.

## Steps for Simulation

### 1. Initialize Simulation
- **Script**: `simulation_main.sh`
- **Purpose**: Set your working directory and output file name.
- **Parameters**:
  - Number of simulations
  - Sample size for each simulation

### 2. Data Generation
- **Script**: `generate_simulation_data.py`
- **Purpose**: Create sequences of observable values and hidden states, and estimate PHMM parameters for these values.

### 3. Forward Probability Estimation
- **Script**: `generate_forward_probability.py`
- **Purpose**: Calculate the forward probability for the final period of the simulation.

### 4. Final Dataset Creation
- **Script**: `generate_final_dataset.py`
- **Purpose**: Compile the final dataset using a specific mathematical formula.

### 5. Welfare Estimation
- **Script**: `simulation_main.R`
- **Purpose**: Assess welfare outcomes based on the targeted policy derived from causal inference methods.

### 6. Result Table Compilation
- **Script**: `generate_final_table.py`
- **Purpose**: Assemble a results table from the outputs of `simulation_main.R`.

### 7. Graph Generation
- **Script**: `generate_graph.py`
- **Purpose**: Produce graphs depicting state-2-probability distributions and cost-welfare relationships for each simulation.
