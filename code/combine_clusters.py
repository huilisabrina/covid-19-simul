#!/usr/bin/env python

#------------------------------------------------------
# CS 205 Final Project
# Combine results across clusters for full graph output

# To be called by parallel_clusters.bash
#------------------------------------------------------

#### =================================
####  PACKAGES SETUP
#### =================================

# Python 2 & 3 compatibility 
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

# Basic packages
import random
import numpy as np
import pandas as pd
import argparse

## Argument parsers
parser=argparse.ArgumentParser(description="\n Combination of results across study clusters")
parser.add_argument('--clusters', default=8, type=str, help='clusters of the disjoint clusters (connected components) in the network graph')
parser.add_argument('--input', metavar='FILE_PATH', type=str, help='File path to the network data that contains the simulation output.')
parser.add_argument('--out', default='sim', metavar='FILE_PREFIX', type=str, help='File path to write the simulation results.')

#######################################################

if __name__ == '__main__':
    args = parser.parse_args()
    col = ['s','e','i','r','h','d']
    temp = {"s": [], "e": [], "i":[], "r": [], "h": [], "d": [], "duration": []}

    # Initialize full graph dataframe
    agg_df = pd.DataFrame(columns=col)

    # Aggregate the final node coutns for each state
    for name in args.clusters.split(","):
        cluster_df = pd.read_csv(args.input + "_" + name +".txt", index_col=False, delim_whitespace=True)
        temp["s"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_s"])
        temp["e"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_e"])
        temp["i"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_i"])
        temp["r"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_r"])
        temp["h"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_h"])
        temp["d"].append(cluster_df.loc[cluster_df.shape[0]-1, "n_d"])
        temp["duration"].append(cluster_df.loc[cluster_df.shape[0]-1, "duration"])

    for i in range(6):
        agg_df.loc[1, col[i]] = sum(temp[col[i]])

    # Maximum duration of the subgraphs is the duration of the full update
    agg_df.loc[1, "duration"] = max(temp["duration"])

    # Save results in csv file
    agg_df.to_csv(args.out+".txt", sep='\t', index=False, na_rep="NA")
