# Usage
This file is for a simulation task for welfare estimation with a targeting method combining both the casual inference method and the Poisson Markov Method

Steps
1. Start with the `simulation_main.sh`: specify your current directory and result file name. Moreover, specify the number of your simulations and the number of samples for each simulation
2. `generate_simulation_data.py`: generate the sequence of observed value and hidden state. Moreover, estimate the PHMM parameters for the generated value
3. `generate_forward_probability.py`: estimate the forward probability for the final period
4. `generate_final_dataset.py`: generate the final data set based on the specified math formula
5. `simulation_main.R`: estimate the welfare based on the targeting policy from the causal inference method
6. `generate_final_table.py`: generate the result table from simulation_main.R' result
7. `generate_graph.py`: generate [state2p distribution graph] and [cost-welfare graph] for each simulation
