import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import argparse

parser = argparse.ArgumentParser(description='Generate final data set')
parser.add_argument('--data_path', type=str, default="", help='path to save the data')
parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations')
parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
args = parser.parse_args()

# Configurations
BASE_PATH = f'{args.data_path}/{args.file_name}'
ITERATIONS = args.num_simulations
COVS = ['X', 'state2p' , 'value_period10', 'predicted_transaction', 'X-state2p', 'X-value_period10', 'X-predicted_transaction']
# ALGORITHMS = ['First Best - X-Learner', 'First Best - Causal Forest']
ALGORITHMS = ['First Best - Causal Forest']
SHIFT = 0.00003
BIN_FACTOR = 2

# Logging configuration
logging.basicConfig(level=logging.INFO)

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created at {path}")

def read_data(file_path):
    """Reads a CSV file and returns a dataframe."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

def plot_welfare_graph_first_best(iteration_id, result_path):
    """Plot and save the welfare graph."""
    # Read data
    uniform = read_data(result_path / f"uniform_policy_result_{iteration_id}.csv")
    first_best = read_data(result_path / f"first_best_policy_{iteration_id}.csv")

    if uniform is None or first_best is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    uniform_ = uniform[uniform["Setting"] != "No Treatment"]
    ax.plot(uniform_["Cost"], uniform_["Welfare"], marker='o', label='Uniform Policy')

    for i, cov in enumerate(COVS):
        for j, al in enumerate(ALGORITHMS):
            first_best_ = first_best[(first_best["Algorithm"] == al) & (first_best["Covariates"] == cov)]
            ax.plot(first_best_["Cost"] + i * SHIFT, first_best_["Welfare"] + i * SHIFT, marker='o', label=f'{al}-{cov}')

    ax.set_xlabel("Cost")
    ax.set_ylabel("Welfare")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.6)

    plt.savefig(result_path / f"first_best_welfare_graph_{iteration_id}.png")
    plt.close()

def plot_welfare_graph_policy_learning(iteration_id, result_path):
    """Plot and save the welfare graph."""
    # Read data
    uniform = read_data(result_path / f"uniform_policy_result_{iteration_id}.csv")
    policy_learning = read_data(result_path / f"policy_learning_result_{iteration_id}.csv")

    if uniform is None or policy_learning is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    uniform_ = uniform[uniform["Setting"] != "No Treatment"]
    ax.plot(uniform_["Cost"], uniform_["Welfare"], marker='o', label='Uniform Policy')

    for i, cov in enumerate(COVS):
        policy_learning_ = policy_learning[policy_learning["Covariates"] == cov]
        ax.plot(policy_learning_["Cost"] + i * SHIFT, policy_learning_["Welfare"] + i * SHIFT, marker='o', label=f'Policy Learning-{cov}')

    ax.set_xlabel("Cost")
    ax.set_ylabel("Welfare")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.6)

    plt.savefig(result_path / f"policy_learning_welfare_graph_{iteration_id}.png")
    plt.close()

def plot_state2p_histogram(iteration_id, result_path):
    """Plot and save the state2p histogram."""
    data = read_data(result_path / f"simulation_data_{iteration_id}.csv")

    if data is None:
        return

    q75, q25 = np.percentile(data["state2p"], [75, 25])
    iqr = q75 - q25
    bin_width = BIN_FACTOR * iqr / (len(data["state2p"]) ** (1/3))

    fig, ax = plt.subplots()
    ax.hist(data["state2p"], bins=int((data["state2p"].max() - data["state2p"].min()) / bin_width))
    ax.set_xlabel("state2p")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of state2p")
    ax.set_xlim(0, 1)
    plt.subplots_adjust(right=0.8)

    plt.savefig(result_path / f"state2p_histogram_{iteration_id}.png")
    plt.close()

def main():
    for iteration_id in range(ITERATIONS):
        file_name = Path(f"{BASE_PATH}/result_{iteration_id}")
        create_directory(file_name)

        plot_welfare_graph_first_best(iteration_id, file_name)
        plot_welfare_graph_policy_learning(iteration_id, file_name)
        plot_state2p_histogram(iteration_id, file_name)

if __name__ == "__main__":
    main()



