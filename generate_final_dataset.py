import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Generate final data set')
parser.add_argument('--data_path', type=str, default="", help='path to save the data')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples')
parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations')
parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
args = parser.parse_args()

def generate_samples(directory, index, num_samples, theta_set="seq_period10"):
    """
    Generate the final dataset by combining generated covariates with real values, states, 
    and forward probabilities, and then calculating Y.

    Parameters:
    directory (str): Directory path where the files are located.
    index (int): Index used to select specific files.
    theta_set (str, optional): Column name to use for the 'theta' variable. Defaults to 'value_period10'.
    """
    print("Start to generate samples...")

    # File Paths
    result_file = f"result_{index}"
    observed_data_path = os.path.join(directory, f"{result_file}/observed_data_{index}.csv")
    hidden_state_path = os.path.join(directory, f"{result_file}/hidden_state_{index}.csv")
    forward_probability_path = os.path.join(directory, f"{result_file}/statep_{index}.csv")

    # Constants
    gamma_1, gamma_2, gamma_3 = 2, 15, 10
    beta_0, beta_1, beta_2, beta_3 = 1, -5, 10, 2

    # Generate covariates
    X = np.random.normal(size=num_samples).reshape(-1, 1) # X ~ N(0, 1)
    D = np.random.choice([0, 1], num_samples).reshape(-1, 1) # D ~ Bernoulli(0.5)
    u = np.random.normal(size=num_samples).reshape(-1, 1) # u ~ N(0, 1)

    try:
        # Import data
        observe_data = pd.read_csv(observed_data_path)
        hidden_state = pd.read_csv(hidden_state_path)
        forward_probability = pd.read_csv(forward_probability_path)
    except FileNotFoundError as e:
        print(f"Error in file reading: {e}")
        return

    # Combine all data
    combined_df = pd.DataFrame({
        'X': X.flatten(),
        'treatment': D.flatten(),
        'u': u.flatten(),
        'state1p': forward_probability['state1p'],
        'state2p': forward_probability['state2p']
    })

    combined_df = pd.concat([combined_df, observe_data, hidden_state], axis=1)

    # Assigning theta
    combined_df['theta'] = combined_df[theta_set]

    # Calculate Y
    combined_df['outcome'] = (beta_0 + beta_1 * combined_df['treatment'] + beta_2 * combined_df['theta'] +
                        beta_3 * combined_df['X'] + gamma_1 * combined_df['treatment'] * combined_df['X'] +
                        gamma_2 * combined_df['treatment'] * combined_df['theta'] +
                        gamma_3 * combined_df['treatment'] * combined_df['X'] * combined_df['theta'])

    print(combined_df.head())

    # Save to CSV
    output_path = os.path.join(directory, f"{result_file}/simulation_data_{index}.csv")
    combined_df.to_csv(output_path, index=False)

    print("Done!!!\n\n")

if __name__ == "__main__":

    directory_path = f'{args.data_path}/{args.file_name}'
    num_samples = args.num_samples
    num_simulations = args.num_simulations

    # run for all 10 samples
    for i in range(num_simulations):
        generate_samples(directory_path, i, num_samples)