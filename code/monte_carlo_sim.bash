#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Wrapper script for Monte Carlo simulation
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir

#### =================================
####  MONTE CARLO SIMULATIONS
#### =================================

# parse Monte Carlo input file
sed '1d' "params_input_test.csv" > MC_param_input.csv

# run Monte Carlo
i=0
while IFS= read -r line; do
    params=($(printf "%s" "$line"|cut -d',' --output-delimiter=' ' -f1-))
    i=$((i+1))
    echo "Begin simulation: $i"
    spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
            network_update_GF_monte_carlo_cluster.py \
            --v_input "v_cluster_4_low.txt" \
            --e_input "e_cluster_4_low.txt" \
            --p_is ${params[0]} \
            --p_id ${params[1]} \
            --p_ih ${params[2]} \
            --p_ir ${params[3]} \
            --p_hr ${params[4]} \
            --p_hd ${params[5]} \
            --t_latent ${params[6]} \
            --t_infectious ${params[7]} \
            --num_i_seeds ${params[8]} \
            --num_time_steps ${params[9]} \
            --out "sim_${i}"
    echo "Finish simulation: $i"
    sleep 3
done < MC_param_input.csv
