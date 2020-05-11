#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Wrapper script for running a single network update on real data
# Benchmarking runtime (speed-up plot)
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir


#### =================================
####  CLUSTER SPEED-UP 
####  2-6 cores, 1-2 threads per core
#### =================================
for c in 2..6; do
    for t in 1 2; do
    spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
                --num-executors $c --executor-cores $t \
                network_update_GF_monte_carlo_cluster.py \
                --v_input "v_n1000.txt" \
                --e_input "e_n1000.txt" \
                --num_i_seeds 100 \
                --num_time_steps 10 \
                --out "sim_test_100i_10t_${c}c_${t}t"
    done
done
