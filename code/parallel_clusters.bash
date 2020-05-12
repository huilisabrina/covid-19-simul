#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Wrapper script for Monte Carlo simulation with cluster parallelization

# The following files must be in the same folder as this script:
# combine_clusters.py
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir


#### ====================================
####  SIMULTANEOUS CLUSTER SIMULATIONS
#### ====================================

# Cluster-level parallelization
for cluster in 1..8; do

    # Define cluster vertex and edge input files
    v_input_name = "v_n1000_cluster_${cluster}.txt"
    e_input_name = "e_n1000_cluster_${cluster}.txt"
    
    echo "Begin cluster: ${cluster}"
    spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
            network_update_GF_monte_carlo.py \
            --v_input ${v_input_name} \
            --e_input ${e_input_name} \
            --num_i_seeds 100 \
            --num_time_steps 10 \
            --out "sim_n1000_cluster_${cluster_name}"
            
done        

# Combine results for full graph
echo "Combining results across clusters"
python combine_clusters.py \
        --clusters "1,2,3,4,5,6,7,8" \
        --input "sim_n1000_cluster" \
        --out "sim_n1000_full"

echo "Finish simulation!"
sleep 3
