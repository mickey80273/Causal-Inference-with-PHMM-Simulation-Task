# Aim: Generate simulation data
# Function: Generate sequences and states data, allowing customization of length and sample size
#          Estimate state probabilities from generated data

from phmm import PHMM
import pandas as pd
import datetime
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Generate final data set')
parser.add_argument('--data_path', type=str, default="", help='path to save the data')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations')
parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
args = parser.parse_args()

def generate_data(transistion_mx, initial_pb, emission_rt, sample_length, num_samples, output_path, idx):

    # Initial parameters
    transition_matrix = transistion_mx
    initial_probabilities = initial_pb
    emission_rates = emission_rt

    # Initialize the model
    print("Initializing the PHMM model...")
    hmm_model = PHMM(initial_probabilities, transition_matrix, emission_rates)

    # Generate sequences and states
    print("Generating sequences and states...")
    sequences, states = zip(*[hmm_model.gen_seq(n=sample_length) for _ in range(num_samples)])

    # Generate column names
    value_columns = ["value_period" + str(1 + i) for i in range(sample_length)]
    seq_columns = ["seq_period" + str(1 + i) for i in range(sample_length)]

    # Save the dataset and states as CSV files
    dataset = pd.DataFrame(sequences, columns=value_columns)
    states_df = pd.DataFrame(states, columns=seq_columns)

    # current_date = datetime.date.today()
    # formatted_date = current_date.strftime("%m-%d-%Y")

    print("Saving the dataset...")
    data_set_name = f'observed_data_{idx}.csv'
    data_dir = f'{output_path}/result_{idx}'
    data_set_path = f'{data_dir}/{data_set_name}'
    # Check if the directory already exists
    if not os.path.exists(data_dir):
        # Create the directory
        os.makedirs(data_dir)
        message = f"Directory '{data_dir}' created."
    else:
        message = f"Directory '{data_dir}' already exists."
    print(message)

    dataset.to_csv(data_set_path, index=False)
    hidden_state_name = f'{data_dir}/hidden_state_{idx}.csv'
    states_df.to_csv(hidden_state_name, index=False)

    print(
    f"Sample size: {num_samples}\n"
    f"Sequence length: {sample_length}\n"
    f"Transition matrix: {transition_matrix}\n"
    f"Initial probability: {initial_probabilities}\n"
    f"Emission Rate: {emission_rates}"
    )
    
    print("Done!!!\n\n")

def estimate_state_probability(output_path, idx):

    # Random Initialization
    transition_matrix_r = [[0.5, 0.5], [0.5, 0.5]]
    initial_probabilities_r = [1.0, 0.0]
    emission_rates_r = [1.0, 5.0]
    hmm_model_random = PHMM(initial_probabilities_r, transition_matrix_r,  emission_rates_r)

    # Import dataset
    data_set_name = f'observed_data_{idx}.csv'
    data_dir = f'{output_path}/result_{idx}'
    data_set_path = f'{data_dir}/{data_set_name}'
    seqs = pd.read_csv(data_set_path)

    # Convert dataset to tuple of sequences
    seqs = tuple(seqs.values.tolist())

    # Estimate state probabilities using Baum-Welch algorithm
    hmm_model_random.baum_welch(seqs)

    # current_date = datetime.date.today()
    # formatted_date = current_date.strftime("%m-%d-%Y")

    print("Saving the dataset...")
    para_name = f'statePHMM_parameters_{idx}.txt'
    data_set_path = f'{data_dir}/{para_name}'
    # Check if the directory already exists
    if not os.path.exists(data_dir):
        # Create the directory
        os.makedirs(data_dir)
        message = f"Directory '{data_dir}' created."
    else:
        message = f"Directory '{data_dir}' already exists."
    print(message)

    # Save estimated parameters to a text file
    with open(data_set_path, 'w+') as f:
        f.write(f'Estimation of 2 state PHMM using monthly transaction data \n \n')
        f.write(f'Transition Matrix: \n {hmm_model_random.transition_matrix()} \n')
        f.write(f'Lambdas:\n {hmm_model_random.lambdas} \n')
        f.write(f'Initial Probability:\n {np.exp(hmm_model_random.delta)} \n')

    print(
        f"\nFollowing are the estimated parameters:\n"
        f"Transition matrix: {hmm_model_random.transition_matrix()}\n"
        f"Initial probability: {hmm_model_random.lambdas}\n"
        f"Emission Rate: {np.exp(hmm_model_random.delta)}\n\n"
    )

if __name__ == "__main__":
    
    output_path = f'{args.data_path}/{args.file_name}'
    num_simulations = args.num_simulations

    # parameters
    transition_matrix = [[0.8, 0.2], [0.9, 0.1]]
    initial_probabilities = [1.0, 0.0]
    emission_rates = [1.0, 10.0]

    for idx in range(num_simulations):
        sample_length = 10
        num_samples = args.num_samples # change your number of samples here

        generate_data(transistion_mx=transition_matrix,
                      initial_pb=initial_probabilities,
                      emission_rt=emission_rates,
                      sample_length=sample_length,
                      num_samples=num_samples, 
                      output_path=output_path,
                      idx= idx)

        # Estimate state probabilities
        estimate_state_probability(output_path, idx)