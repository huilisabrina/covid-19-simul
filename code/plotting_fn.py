#===========================================
# Plotting Functions for Simulation Output
#
# Must be in same folder as output files
#===========================================

import os
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt

os.chdir("C:/Users/Lang/Downloads")

## Monte Carlo Plotting Function
## Plots the number of S, E, I, R, H, and D nodes as a function of time.
def make_plot(df, param_df, index, num_nodes=596):
    plt.figure()
    num_time_steps = len(df) - 1 
    param = param_df.iloc[index]
    plt.title("Monte Carlo with n_s=" + str(param[0])+", t_latent="+str(param[6])+", t_infect="+str(param[7])+
              "\n p_id=0.0134, p_ih=0.0678, p_ir=0.3945, p_hr=0.19725, \np_hd=0.0419, i_seeds=20, max_time=20")
    h1, = plt.plot(df['n_s'] / num_nodes, color="red", label='S')
    h2, = plt.plot(df['n_e'] / num_nodes, color="orange", label='E')
    h3, = plt.plot(df['n_i'] / num_nodes, color="green", label='I')
    h4, = plt.plot(df['n_r'] / num_nodes, color="blue", label='R')
    h5, = plt.plot(df['n_h'] / num_nodes, color="purple", label='H')
    h6, = plt.plot(df['n_d'] / num_nodes, color="black", label='D')
    plt.xlabel("Time")
    plt.xlim(0, num_time_steps)
    plt.legend(loc=5)
    #plt.show()
    plt.savefig('mc_plot_'+ str(index+1)+'.png')

## Connectivity Graph Plotting Function
## Plots the number of S, E, I, R, H, and D nodes as a function of time.
def make_plot_connect(df, title, connect, num_nodes=596):
    plt.figure()
    num_time_steps = len(df)
    plt.title(title)
    h1, = plt.plot(df['n_s'] / num_nodes, color="red", label='S')
    h2, = plt.plot(df['n_e'] / num_nodes, color="orange", label='E')
    h3, = plt.plot(df['n_i'] / num_nodes, color="green", label='I')
    h4, = plt.plot(df['n_r'] / num_nodes, color="blue", label='R')
    h5, = plt.plot(df['n_h'] / num_nodes, color="purple", label='H')
    h6, = plt.plot(df['n_d'] / num_nodes, color="black", label='D')
    plt.xlabel("Time")
    plt.xlim(0, num_time_steps)
    plt.legend(loc=5)
    #plt.show()
    plt.savefig('connect_plot_'+ connect +'.png')

# Load in dataset of MC paramenters, where row number corresponds to run number
params = pd.read_csv('MC_param_input.csv', header=None)

# Load in results of MC runs
data_1 = pd.read_csv('sim_cluster_4_med_1.txt', sep='\t', header=0)
data_2 = pd.read_csv('sim_cluster_4_med_2.txt', sep='\t', header=0)
data_4 = pd.read_csv('sim_cluster_4_med_4.txt', sep='\t', header=0)
data_4 = data_4.dropna()
data_7 = pd.read_csv('sim_cluster_4_med_7.txt', sep='\t', header=0)
data_7 = data_7.dropna()
data_8 = pd.read_csv('sim_cluster_4_med_8.txt', sep='\t', header=0)
data_8 = data_8.dropna()
data_9 = pd.read_csv('sim_cluster_4_med_9.txt', sep='\t', header=0)
data_9 = data_9.dropna()
data_10 = pd.read_csv('sim_cluster_4_med_10.txt', sep='\t', header=0)
data_10 = data_10.dropna()
data_11 = pd.read_csv('sim_cluster_4_med_11.txt', sep='\t', header=0)
data_11 = data_11.dropna()
data_12 = pd.read_csv('sim_cluster_4_med_12.txt', sep='\t', header=0)
data_12 = data_12.dropna()

# Make plots for Monte Carlo simulations
make_plot(df = data_1, param_df = params, index = 0)
make_plot(df = data_2, param_df = params, index = 1)
make_plot(df = data_4, param_df = params, index = 3)
make_plot(df = data_7, param_df = params, index = 6)
make_plot(df = data_8, param_df = params, index = 7)
make_plot(df = data_9, param_df = params, index = 8)
make_plot(df = data_10, param_df = params, index = 9)
make_plot(df = data_11, param_df = params, index = 10)
make_plot(df = data_12, param_df = params, index = 11)

# Load in connectivity datasets
data_low = pd.read_csv('connectivity_low_cluster_4_100i_10t.txt', sep='\t', header=0)
data_med = pd.read_csv('connectivity_med_cluster_4_100i_10t.txt', sep='\t', header=0)
data_high = pd.read_csv('connectivity_all_cluster_4_100i_10t.txt', sep='\t', header=0)

# Make plots for connectivity simulations
make_plot_connect(df = data_low, title = "Low Connectivity with 596 Nodes and 44327 Edges \n p_is = 0.5, p_id = 0.02, p_ih=0.06, p_ir=0.3, p_hr=0.15, "+
"\np_hd=0.04, t_latent=5, t_infect=5, i_seeds=100, max_time=10", connect = "low")
make_plot_connect(df = data_med, title = "Med Connectivity with 596 Nodes and 132982 Edges\n p_is = 0.5, p_id = 0.02, p_ih=0.06, p_ir=0.3, p_hr=0.15, "+
"\np_hd=0.04, t_latent=5, t_infect=5, i_seeds=100, max_time=10", connect = "med")
make_plot_connect(df = data_high, title = "High Connectivity with 596 Nodes and 177310 Edges\n p_is = 0.5, p_id = 0.02, p_ih=0.06, p_ir=0.3, p_hr=0.15, "+
"\np_hd=0.04, t_latent=5, t_infect=5, i_seeds=100, max_time=10", connect = "high")