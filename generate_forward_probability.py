# Aim: generate the forward probability
# Steps:
# 1. Get the estimated parameters with baum welch algorithm
# 2. Calculate the forward probability in each data

import re
import os
import pandas as pd
from phmm import PHMM
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate final data set')
parser.add_argument('--data_path', type=str, default="", help='path to save the data')
parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations')
parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
args = parser.parse_args()

def extract_parameters(contents):
    regex_patterns = {
        'transition_matrix': r"Transition Matrix:\s*\[\[([\d.]+)\s*([\d.]+)\s*\]\s*\[([\d.]+)\s*([\d.]+)\s*\]\]",
        'lambdas': r"Lambdas:\s*\[\s*([\d.]+)\s+([\d.]+)\s*\]",
        'initial_probability': r"Initial Probability:\s*\[([\d.]+) ([\d.]+)\]"
    }
    parameters = {}

    for param, regex in regex_patterns.items():
        matches = re.search(regex, contents)
        if matches:
            parameters[param] = [float(value) for value in matches.groups()]
        else:
            raise ValueError(f"Parameter '{param}' not found in the file.")

    return parameters

def process_file(para_path, observed_data_path, output_path):
    with open(para_path, 'r') as file:
        contents = file.read()

    params = extract_parameters(contents)

    transition_matrix = np.array(params['transition_matrix']).reshape(2, 2)
    lambdas = params['lambdas']
    initial_probability = params['initial_probability']

    df = pd.read_csv(observed_data_path)
    sequences = [sublist[:10] for sublist in df.values.tolist()]

    phmm = PHMM(initial_probability, transition_matrix, lambdas)

    final_probabilities = []
    for sequence in tqdm(sequences, total=len(sequences)):
        forward_probs = phmm.forward_lprobs(sequence)
        final_probabilities.append(forward_probs[-1])

    df_final = pd.DataFrame(final_probabilities, columns=['state1p', 'state2p'])
    df_final = df_final.div(df_final.sum(axis=1), axis=0)
    
    df_final.to_csv(output_path)

if __name__ == "__main__":
    
    directory_path = f'{args.data_path}/{args.file_name}'
    num_simulations = args.num_simulations

    for idx in range(num_simulations):
        data_dir = os.path.join(directory_path, f'result_{idx}')
        para_path = os.path.join(data_dir, f'statePHMM_parameters_{idx}.txt')
        observed_data_path = os.path.join(data_dir, f'observed_data_{idx}.csv')
        output_path = os.path.join(data_dir, f'statep_{idx}.csv')

        print(f"Processing: {data_dir}")
        try:
            process_file(para_path, observed_data_path, output_path)
        except Exception as e:
            print(f"Error processing {data_dir}: {e}")
