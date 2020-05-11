#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Wrapper script for Monte Carlo simulation with cluster parallelization

# The following files must be in the same folder as this script:
#   preprocess_network.py
#   edge_list.csv
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir

# preprocess the HIV datasets
python preprocess_network.py

#### =================================
####  SINGLE RUN TEST
#### =================================
# run simulation pipeline (default parameters)
spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
            network_update_GF_monte_carlo_local.py \
            --v_input "v_n1000.txt" \
            --e_input "e_n1000.txt" \
            --num_i_seeds 100 \
            --num_time_steps 10 \
            --out "sim_n1000"
