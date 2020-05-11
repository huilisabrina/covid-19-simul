#!/bin/bash

#------------------------------------------------------------------------
# CS 205 Final Project
# Hypothesis testing: High, medium and low connectivity of the graph
#------------------------------------------------------------------------

wd_dir="/home/ubuntu/CS205_FinalProject/testing"
cd $wd_dir

for type in all low med; do
    hadoop fs -put v_cluster_4_$type.txt
    hadoop fs -put e_cluster_4_$type.txt
done

#### ======================================
####  wHAT DOES CONNECTIVITY OF GRAPHS DO?
#### ======================================
for type in all low med; do
    spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 \
                --num-executors 2 --executor-cores 2 \
                network_update_GF_monte_carlo_cluster.py \
                --v_input "v_cluster_4_$type.txt" \
                --e_input "e_cluster_4_$type.txt" \
                --num_i_seeds 100 \
                --num_time_steps 10 \
                --out "connectivity_${type}_cluster_4_100i_10t" &
done
