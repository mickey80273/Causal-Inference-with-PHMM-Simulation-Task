#!/bin/sh
# change the environment
# conda activate targeting_simulation

# Simulation Task
# Step:
#     1. generate data with PHMM and store the observed data and hidden state data, and estiamte the parameters
#     2. generate forward probability
#     3. generate the final dataset based on the math equation
#     4. run the R script to estimate the treatment effect
#     5. generate the final table
#     6. draw the graph

# Remember to change the path of the python file
dir_path="/home/u56101022/Project/SimulationTask"
file_name="v0_20240109"
num_simulation=2
num_samples=100

# Generate and store the observced data and hidden state data; estimate the parameters
python3 generate_simulation_data.py --data_path $dir_path --file_name $file_name --num_samples $num_samples --num_simulation $num_simulation
## PS: you can change the parameters of PHMM in the python file

# Generate transaction prediction
python3 PurchasePredictionModel/run.py --data_path $dir_path --num_simulation $num_simulation  --file_name $file_name \
                                        --num_epochs 100 --batch_size 100 --learning_rate 0.001 

# ssh -x node01 # please change the node number
# Generate the forward probability
python3 generate_forward_probability.py --data_path $dir_path --file_name $file_name --num_simulation $num_simulation

# Generate the final dataset based on the math equation
python3 generate_final_dataset.py --data_path $dir_path --file_name $file_name --num_samples $num_samples --num_simulation $num_simulation

# Run the R script to estimate the treatment effect
Rscript simulation_main.R $dir_path $file_name $num_simulation

# Generate the fianl table
python3 generate_final_table.py --data_path $dir_path --file_name $file_name --num_simulation $num_simulation

# Draw the graph of cost
python3 generate_graph.py --data_path $dir_path --file_name $file_name --num_simulation $num_simulation