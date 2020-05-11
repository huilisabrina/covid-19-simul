#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Wrapper script for Monte Carlo simulation with cluster parallelization

# The following files must be in the same folder as this script:
#   preprocess_network.py
#   params_input.csv
#   edge_list.csv
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir

# preprocess the HIV datasets
python preprocess_network.py

# suppress INFO messages displaying in Spark console:
# spark-submit \
# --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
# --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:/home/ubuntu/spark-2.2.0-bin-hadoop2.7/conf/log4j.properties.template" \
# --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:/home/ubuntu/spark-2.2.0-bin-hadoop2.7/conf/log4j.properties.template" \
#  network_update_GF.py
# ^ NO NEED IF WE DIRECTLY CHANGE THE LOG LEVEL OF SPARK CONTEXT



#### ====================================
####  SIMULTANEOUS CLUSTER SIMULATIONS
#### ====================================

sim_counter=0
while IFS= read -r line;do
    params=($(printf "%s" "$line"|cut -d',' --output-delimiter=' ' -f1-))
    sim_counter = sim_counter + 1
    echo "Begin simulation: $sim_counter"

    # Cluster-level parallelization
    for cluster in 4; do

        # Define cluster vertex and edge input files
        v_input_name = "v_orig_cluster_${cluster}.txt"
        e_input_name = "e_orig_cluster_${cluster}.txt"
        
        echo "Begin cluster: ${cluster}"
        spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 network_update_GF_monte_carlo.py \
                --v_input ${v_input_name} \
                --e_input ${e_input_name} \
                --p_is ${params[0]} \
                --p_id ${params[1]} \
                --p_ih ${params[2]} \
                --p_ir ${params[3]} \
                --p_hr ${params[4]} \
                --p_hd ${params[5]} \
                --t_latent ${params[6]} \
                --t_infectious ${params[7]} \
                --num_i_seeds ${params[8} \
                --num_s_seeds ${params[9]} \
                --num_h_seeds ${params[10]} \
                --num_time_steps ${params[11]} \
                --out "sim_${sim_counter}_cluster_${cluster_name}"
                
    done        

    # Combine results for full graph
    echo "Combining results across clusters"
    python combine_clusters.py \
            --clusters "1,2,3" \
            --input "sim_${sim_counter}_cluster" \
            --out "sim_${sim_counter}_full"

    echo "Finish simulation: $sim_counter"
    sleep 3
    
done < MC_param_input.csv