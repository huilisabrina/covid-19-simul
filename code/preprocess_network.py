#!/usr/bin/env python

#-------------------------------------------------------
# CS 205 Final Project
# Preprocess the HIV dataset for network updates

# The following files must be in the same folder as this script:
#   edge_list.csv
#-------------------------------------------------------

import random
import numpy as np
import pandas as pd
import itertools as it

def gen_v_and_e(df):

    # Construct the vertices table
    id_list_1 = list(df['ID1'])
    id_list_2 = list(df['ID2'])

    temp = list(set(id_list_1) | set(id_list_2))
    id_list = [x.split("_")[0] for x in temp]
    cluster_list = [x.split("_")[1] for x in temp]
    v_df = pd.DataFrame(data={"id": id_list, "cluster": cluster_list})
    v_df = v_df[["id", "cluster"]]

    # Construct the edges table
    e_df = df[["ID1", "ID2", "STUDYNUM"]]
    e_df["src"] = e_df["ID1"].str.slice(start=0, stop=-2, step=1)
    e_df["dst"] = e_df["ID2"].str.slice(start=0, stop=-2, step=1)
    e_df = e_df[["src","dst"]]

    return v_df, e_df

def gen_v(df_e):

    id_list_1 = list(df_e['src'])
    id_list_2 = list(df_e['dst'])

    id_list = list(set(id_list_1) | set(id_list_2))
    v_df = pd.DataFrame(data={"id": id_list})

    return v_df

#### =======================================
####  PREPARE NETWORK DATASETS
#### =======================================

# Load raw (cleaned) network data (edge_list.csv is the network_data.csv)
network_data="edge_list.csv"
df = pd.read_csv(network_data, sep=",", index_col=False)

# all clusters together (original)
v_df, e_df = gen_v_and_e(df)
v_df.to_csv("v_orig.txt", sep='\t', index=False, na_rep="NA")
e_df.to_csv("e_orig.txt", sep='\t', index=False, na_rep="NA")

# use a random 1000 edges to work with
df_n1000 = df.sample(n=1000, random_state=1)
v_df, e_df = gen_v_and_e(df_n1000)
v_df.to_csv("v_n1000.txt", sep='\t', index=False, na_rep="NA")
e_df.to_csv("e_n1000.txt", sep='\t', index=False, na_rep="NA")

# Split up data by study number, defined through "cluster" column
for cluster in range(1,9):

    df_cluster = df.loc[df['STUDYNUM'] == cluster]

    v_df, e_df = gen_v_and_e(df_cluster)
    v_df.to_csv("v_orig_cluster_" + str(cluster) + ".txt", sep='\t', index=False, na_rep="NA")
    e_df.to_csv("e_orig_cluster_" + str(cluster) + ".txt", sep='\t', index=False, na_rep="NA")

    print("Cluster: {} \n Number of vertices: {} \n Number of edges: {}".format(cluster, v_df.shape[0], e_df.shape[0]))

# Modify the connectivity of the graph (simulated data for testing assumptions)
cluster = 4
df_cluster = df.loc[df['STUDYNUM'] == cluster]
v_df, _ = gen_v_and_e(df_cluster)

all_comb = list(it.combinations(list(v_df["id"]), 2))
e_df_all = pd.DataFrame(data={"src": [x[0] for x in all_comb], "dst": [x[1] for x in all_comb]})
v_df.to_csv("v_cluster_" + str(cluster) + "_all.txt", sep='\t', index=False, na_rep="NA")
e_df_all.to_csv("e_cluster_" + str(cluster) + "_all.txt", sep='\t', index=False, na_rep="NA")

e_df_med = e_df_all.sample(n=int(e_df_all.shape[0]*0.75), random_state=1)
v_df_med = gen_v(e_df_med)
v_df.to_csv("v_cluster_" + str(cluster) + "_med.txt", sep='\t', index=False, na_rep="NA")
e_df_med.to_csv("e_cluster_" + str(cluster) + "_med.txt", sep='\t', index=False, na_rep="NA")

e_df_low = e_df_all.sample(n=int(e_df_all.shape[0]*0.25), random_state=1)
v_df_low = gen_v(e_df_low)
v_df.to_csv("v_cluster_" + str(cluster) + "_low.txt", sep='\t', index=False, na_rep="NA")
e_df_low.to_csv("e_cluster_" + str(cluster) + "_low.txt", sep='\t', index=False, na_rep="NA")
