#!/usr/bin/env python

#-------------------------------------------------------
# CS 205 Final Project
# Preprocess the HIV dataset 
# Prepare network input dataset (incorporate assumptions)
#-------------------------------------------------------

import random
import numpy as np
import pandas as pd

# Load raw (cleaned) network data (edge_list.csv is the network_data.csv)
network_data="edge_list.csv"
df = pd.read_csv(network_data, sep=",", index_col=False)

# Randomly select 100 lines from the original dataset for testing
df = df.sample(n=100, random_state=1)

#### =======================================
####  BASIC PREP (USE DATASET AT FACE VALUE)
#### =======================================

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

# Save to files
v_df.to_csv("v_orig.txt", sep='\t', index=False, na_rep="NA")
e_df.to_csv("e_orig.txt", sep='\t', index=False, na_rep="NA")

#### =======================================
####  CHANGE NETWORK AT OUR DISCRETION
#### =======================================

# CHANGE THE SIZE OF THE NETWORK DATA...

# ADD RANDOM EDGES....

