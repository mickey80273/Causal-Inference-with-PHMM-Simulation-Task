'''
Aim: generate final analysis table
Step:
    1. select the cost
    2. create the table for uniform policy
    3. create the table for first_best policy and policy learning
'''

from tabulate import tabulate
import pandas as pd
import os
import argparse

def get_covariate_data(f_b, covariate_name):
            default_values = {"Welfare": "N/A", "Std": "N/A", "Treat": "N/A", "Non_treat": "N/A"}

            # Filter the DataFrame for the specific covariate
            filtered_data = f_b.loc[f_b["Covariates"] == covariate_name, ["Welfare", "Std", "Treat", "Non_treat"]]

            # If no data is found, use default values
            if filtered_data.empty:
                data = default_values
            else:
                # Assuming the first row is the desired data
                data = filtered_data.iloc[0].fillna(default_values)

            # Creating the formatted string
            welfare, std, treat, ntreat = [data[key] for key in ["Welfare", "Std", "Treat", "Non_treat"]]

            # Check if welfare and ntreat can be converted to float, and format accordingly
            try:
                welfare_formatted = f"{float(welfare):.4f}" if welfare != "N/A" else welfare
                std_formatted = f"{float(std):.4f}" if std != "N/A" else std
                treat_formatted = f"{float(treat):.4f}" if treat != "N/A" else treat
                ntreat_formatted = f"{float(ntreat):.4f}" if ntreat != "N/A" else ntreat
            except ValueError:
                welfare_formatted = welfare
                std_formatted = std
                treat_formatted = treat
                ntreat_formatted = ntreat

            w_s = f'{welfare_formatted}\n({std_formatted})\n{treat_formatted} / {ntreat_formatted}'

            return {
                "welfare_str": welfare,
                "std_str": std,
                "treat_str": treat,
                "ntreat_str": ntreat,
                "w_s": w_s
            }

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate final data set')
    parser.add_argument('--data_path', type=str, default="", help='path to save the data')
    parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations')
    parser.add_argument('--file_name', type=str, default="", help='the file name to save the data')
    args = parser.parse_args()

    file_ = f'{args.data_path}/{args.file_name}'
    covs = ['X', 'state2p' , 'value_period10', 'predicted_transaction', 'X-state2p', 'X-value_period10', 'X-predicted_transaction']
    # algorithms = ['First Best - X-Learner', 'First Best - Causal Forest']
    algorithms = ['First Best - Causal Forest']
    cost = [0, 0.5, 0.7, 0.9]

    num_simulations = args.num_simulations

    for iteration_id in range(num_simulations):
        base_path = f"{file_}/result_{iteration_id}"
        uniform_path = f"{base_path}/uniform_policy_result_{iteration_id}.csv"
        first_best_path = f"{base_path}/first_best_policy_{iteration_id}.csv"
        policy_learning_path = f"{base_path}/policy_learning_result_{iteration_id}.csv"

        uniform = pd.read_csv(uniform_path)
        first_best = pd.read_csv(first_best_path)
        policy_learning = pd.read_csv(policy_learning_path)

        for c in cost:      
            uniform_ = uniform[uniform["Cost"]==c]
            first_best_ = first_best[first_best["Cost"]==c]
            policy_learning_ = policy_learning[policy_learning["Cost"]==c]

            first_best_ls = []
            for al in algorithms:
                f_b = first_best_[first_best_["Algorithm"]==al]
                X_data = get_covariate_data(f_b, "X")
                state2p_data = get_covariate_data(f_b, "state2p")
                value_period10_data = get_covariate_data(f_b, "value_period10")
                X_state2p_data = get_covariate_data(f_b, "X-state2p")
                X_value_period10_data = get_covariate_data(f_b, "X-value_period10")
                append_data = [al, 
                                X_data["w_s"], state2p_data["w_s"], value_period10_data["w_s"], 
                                X_state2p_data["w_s"], X_value_period10_data["w_s"]]
                first_best_ls.append(append_data)

            pl = policy_learning_
            pl_X_data = get_covariate_data(pl, "X")
            pl_state2p_data = get_covariate_data(pl, "state2p")
            pl_value_period10_data = get_covariate_data(pl, "value_period10")
            pl_X_state2p_data = get_covariate_data(pl, "X-state2p")
            pl_state2p_dataX_value_period10_data = get_covariate_data(pl, "X-value_period10")
            pl_data = [["Policy Learning", 
                        pl_X_data["w_s"], pl_state2p_data["w_s"], pl_value_period10_data["w_s"], 
                        pl_X_state2p_data["w_s"], pl_state2p_dataX_value_period10_data["w_s"]]]

            file_path = os.path.join(base_path, 'result_table_' + str(iteration_id) + '.txt')

            try:
                with open(file_path, 'a') as f:
                    # Uniform policy
                    welfare = uniform_["Welfare"].iloc[0]
                    std = uniform_["Std"].iloc[0]
                    treat = uniform_["Treat"].iloc[0]
                    non_treat = uniform_["Non_treat"].iloc[0]

                    uniform_n = uniform[uniform["Cost"]==0]
                    welfare_n = uniform_n["Welfare"].iloc[0]
                    std_n = uniform_n["Std"].iloc[0]
                    treat_n = uniform_n["Treat"].iloc[0]
                    non_treat_n = uniform_n["Non_treat"].iloc[0]

                    f.write(f"Cost: {c*100}%\n")
                    
                    f.write(f"Welfare from uniform policy\n")
                    # All treatment
                    f.write("\nAll Treatment:\n")
                    f.write(f"Welfare: {welfare:.4f} ({std:.4f})\n")
                    f.write(f"Treat / Non-treat: {treat}/{non_treat}\n")
                    # No treatment
                    f.write("\nNo Treatment:\n")
                    f.write(f"Welfare: {welfare_n:.4f} ({std_n:.4f})\n")
                    f.write(f"Treat / Non-treat: {treat_n}/{non_treat_n}\n")

                    # First best policy
                    f.write(f"\nWelfare from First Best Policy\n")
                    f.write(tabulate(first_best_ls, headers=covs, tablefmt="grid"))

                    # Policy learning
                    f.write(f"\nWelfare from Policy Learning\n")
                    f.write(tabulate(pl_data, headers=covs, tablefmt="grid"))
                    f.write("\n\n")

            except IOError as e:
                print(f"An error occurred while writing to the file: {e}")
