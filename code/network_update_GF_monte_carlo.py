#!/usr/bin/env python

#-------------------------------------------------------
# CS 205 Final Project
# Network updating functions with GF integrated

# Major differences from network_update_GF
# Fully integrated with Command line
# To be called by monte_carlo.bash
# Replace toy example with real data
#-------------------------------------------------------

#### INITIATE SPARK WITH THE FOLLOWING
# pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11

#### CALL FROM COMMAND LINE
# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 network_update_GF.py

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
from functools import reduce
import warnings
warnings.filterwarnings("ignore")
import os, sys, re
import logging, time, traceback
import argparse
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark-shell")
# import scipy.stats as ss
# import matplotlib.pyplot as plt
# %matplotlib notebook

# graphframes
from graphframes import *
from graphframes import graphframe as GF
from graphframes.lib import AggregateMessages as AM
from graphframes.examples import Graphs

# MPI (later)
# from mpi4py import MPI

# SQL + Spark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, lit, udf, when, concat, collect_list
from pyspark.sql.functions import sum as fsum
from pyspark.sql.types import *


####### SET UP Spark environment
# Spark configuration
conf = SparkConf().setMaster('local[2]').setAppName('local_run')
# Spark context
sc = SparkContext(conf=conf)
# surpress logging
sc.setLogLevel("ERROR")
# Create an SQL context:
sql_context = SQLContext(sc)


#### =================================
####  HELPER FUNCTIONS
#### =================================

class Logger_to_Logging(object):
    '''
    Logger class that write uses logging module and is needed to use munge_sumstats or ldsc from the LD score package.
    '''
    def __init__(self):
        logging.info('created Logger instance to pass through ldsc.')
        super(Logger_to_Logging, self).__init__()

    def log(self,x):
        logging.info(x)

class DisableLogger():
    '''
    For disabling the logging module when calling munge_sumstats
    '''
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, a, b, c):
       logging.disable(logging.NOTSET)

def safely_create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        if not os.path.isdir(folder_path):
            raise

def sec_to_str(t):
    
    '''Convert seconds to days:hours:minutes:seconds'''
    
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f
    
def construct_graph(args):

    # Load pre-processed datasets
    v = pd.read_csv(args.v_input, index_col=False, delim_whitespace=True)
    e = pd.read_csv(args.e_input, index_col=False, delim_whitespace=True)

    logging.info("Setting up graph data with {} nodes, {} edges and {} clusters".format(v.shape[0], e.shape[0], len(v["cluster"].unique())))

    # Coin dataframes into SQL for graphframes
    v_schema = StructType([StructField("id", IntegerType(), True),
                           StructField("cluster", IntegerType(), True)])
    v = sql_context.createDataFrame(v, schema = v_schema).dropDuplicates(['id'])

    e_schema = StructType([StructField("src", IntegerType(), True), 
                        StructField("dst", IntegerType(), True)])
    e = sql_context.createDataFrame(e, schema = e_schema)

    # Generate graph before simulation starts
    g = GF.GraphFrame(v, e)
    
    return g

def save_sim(args, nodes_counter, duration):

    # write population size table
    nodes_counter.to_csv(args.out+".txt", sep='\t', index=False, na_rep="NA")

    # write duration and parameters
    summary = "\n Summary of simulation: \n"
    summary += "------------------------------------- \n"
    summary += "Duration: " + str(duration) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_is: " + str(args.p_is) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_id: " + str(args.p_id) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_ih: " + str(args.p_ih) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_ir: " + str(args.p_ir) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_hr: " + str(args.p_hr) + '\n'
    summary += "------------------------------------- \n"
    summary += "p_hd: " + str(args.p_hd) + '\n'
    summary += "------------------------------------- \n"
    summary += "t_latent: " + str(args.t_latent) + '\n'
    summary += "------------------------------------- \n"
    summary += "t_infectious: " + str(args.t_infectious) + '\n'

    # save summary
    logging.info(summary)

    return 0
    
#### =================================
####  GRAPH UPDATING FUNCTIONS
#### =================================

#Carry out the H -> R,D process for one time step
def H_flow(h_nodes, r_nodes, d_nodes, p_hr, p_hd):

    #Input check
    assert 1-p_hr-p_hd > 0, "Unrealistic p_hr and p_hd input."

    new_recoveries = []
    new_deaths = []

    if len(h_nodes) == 0:
        return(0)

    for node in h_nodes:
        dst_state = np.random.multinomial(1, [p_hr, p_hd, 1-p_hr-p_hd])
        dst_index = np.argwhere(dst_state==1)[0,0]
        if dst_index == 0: #Flow to recovery
            r_nodes.append(node)
            new_recoveries.append(node)
        elif dst_index == 1: #Flow to death
            d_nodes.append(node)
            new_deaths.append(node)
        #else stay at H   
    
    #Remove nodes that have changed state
    for node in new_recoveries + new_deaths:
        h_nodes.remove(node)
    
    return 0

#Carry out the I -> D,H,R process for one time step.
def I_flow(i_nodes, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir):

    #Input check
    assert 1-p_id-p_ih-p_ir > 0, "Unrealistic p_id, p_ih and p_ir input."

    new_recoveries = []
    new_hospitalized = []
    new_deaths = []

    for node in i_nodes:
        dst_state = np.random.multinomial(1, [p_id, p_ih, p_ir, 1-p_id-p_ih-p_ir])
        dst_index = np.argwhere(dst_state==1)[0,0]
        if dst_index == 0: #Flow to death
            d_nodes.append(node)
            new_deaths.append(node)
        elif dst_index == 1: #Flow to hospitalized
            h_nodes.append(node)
            new_hospitalized.append(node)
        elif dst_index == 2: #Flow to recovery
            r_nodes.append(node)
            new_recoveries.append(node)
        # else stay as I
    
    #Remove nodes that have changed state
    for node in new_recoveries + new_hospitalized + new_deaths:
        i_nodes.remove(node)
    
    return 0

#Carry out the E -> I process for one time step.
def E_flow(g, e_nodes, i_nodes, t_latent):

    new_I_nodes = []

    for node in e_nodes:
        # extract e_days from the graph
        e_days = g.vertices.filter(col("id").isin([node])).select("e_days").toPandas()["e_days"][0]

        # an exposed individual becomes infectious after t_latent period has been reached
        if e_days == t_latent:
            i_nodes.append(node)
            new_I_nodes.append(node)

    return new_I_nodes

#Carry out the I --> S --> E spreading process for one time step.
def S_flow(g, s_nodes, e_nodes, i_nodes, r_nodes, h_nodes, d_nodes, p_is, p_id, p_ih, p_ir, t_infectious):

    new_E_nodes = []
    for node in i_nodes:
        # extract i_days from the graph
        i_days = g.vertices.filter(col("id").isin([node])).select("i_days").toPandas()["i_days"][0]

        # an I node can only continue to infect people if t_infectious has not been reached
        if i_days <= t_infectious:

            #Select neighbors (only infect via existing edges)
            neighbors = g.vertices.filter(col("id").isin([node])).select("neighbors").toPandas()["neighbors"][0]
            
            if neighbors is None or len(neighbors) == 0: #Assuming network stays constant, a I node w/o neighbors flow to R,D,H
                i_nodes.remove(node)
                send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)
            else:
                infected = random.choice(neighbors)
                if infected in s_nodes: #One infection per person per period
                    flip = np.random.binomial(1, p_is)
                    if flip == 1: #Infect others
                        new_E_nodes.append(infected)
                        e_nodes.append(infected)
                        s_nodes.remove(infected) 

        else: # if I node is no longer infectious (i.e. i_days > t_infectious), go to one of R,H,D
            i_nodes.remove(node)
            send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)

    return new_E_nodes

#Send I node to one of D,H,R once infectious period is reached
def send_I(i_node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir):

    #Reweight p_id, p_ih, p_ir so that transition is definitive
    ls_sum = sum([p_id, p_ih, p_ir])
    p_id, p_ih, p_ir = [x/ls_sum for x in [p_id, p_ih, p_ir]]
    # assert p_ir == 1-p_id-p_ih, "Wrong reweighting of p_id, p_ih and p_ir!"

    dst_state = np.random.multinomial(1, [p_id, p_ih, 1-p_id-p_ih])
    dst_index = np.argwhere(dst_state==1)[0,0]
    if dst_index == 0: #Flow to death
        d_nodes.append(i_node)
    elif dst_index == 1: #Flow to hospitalized
        h_nodes.append(i_node)
    else: #Flow to recovery
        r_nodes.append(i_node)
    return 0

#### =================================
####  SIMULATION WRAPPER
#### =================================

#Simulate an epidemic
def simulate(g, 
             p_is, p_id, p_ih, p_ir, p_hr, p_hd, 
             t_latent, t_infectious, 
             num_i_seeds, num_time_steps):

    # select the vertices as a list
    nodes = list(g.vertices.select("id").toPandas()["id"])
    
    i_nodes = list(random.sample(nodes, num_i_seeds))
    s_nodes = list(set(nodes) - set(i_nodes))
    h_nodes = []
    e_nodes = []
    r_nodes = []
    d_nodes = []
    duration = 0

    nodes_counter = pd.DataFrame(columns=["n_{}".format(x) for x in ['s','e','i','r','h','d']], index=[i for i in range(num_time_steps)])
    nodes_counter.loc[0] = [len(s_nodes),len(e_nodes),len(i_nodes), len(r_nodes), len(h_nodes), len(d_nodes)]

    # ADD "neighbors" column: select neighbors of a node based on a graph (used in S_flow step)
    neighbor = g.find("(a)-[e]->(b)").drop("e").groupBy('a.id').agg(collect_list('b.id').alias('neighbors'))
    g_neighbor = neighbor.join(g.vertices, ['id'], "right_outer")
    g = GF.GraphFrame(g_neighbor, g.edges)

    # ADD "state" column
    # At t0: number of I nodes and H nodes are based on user-defined functions. ALL OTHER NODES are assumed to be S
    g_temp = g.vertices.withColumn("state",when(g.vertices.id.isin(i_nodes), "I").otherwise(when(g.vertices.id.isin(h_nodes), "H").otherwise("S")))
    
    # ADD "i_days" and "e_days" column 
    # At t0: 1 for all I_nodes and 0 for all others
    g_temp = g_temp.withColumn("i_days", lit(0))
    g_temp = g_temp.withColumn("e_days", when(g.vertices.id.isin(e_nodes), lit(1)).otherwise(lit(0)))
    g = GF.GraphFrame(g_temp, g.edges)

    # TO DO: allow p_is to be a vector of rates (time-dependent)
    for step in range(1, num_time_steps+1):
        H_flow(h_nodes, r_nodes, d_nodes, p_hr, p_hd)
        I_flow(i_nodes, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)
        new_I_nodes = E_flow(g, e_nodes, i_nodes, t_latent)
        new_E_nodes = S_flow(g, s_nodes, e_nodes, i_nodes, r_nodes, h_nodes, d_nodes, p_is, p_id, p_ih, p_ir, t_infectious)
                   
        # update the state column using the new lists ("x_nodes")
        g_temp = g.vertices.withColumn("state", when(g.vertices.id.isin(s_nodes), "S").otherwise(when(g.vertices.id.isin(e_nodes), "E").otherwise(when(g.vertices.id.isin(i_nodes), "I").otherwise(when(g.vertices.id.isin(r_nodes), "R").otherwise(when(g.vertices.id.isin(h_nodes), "H").otherwise("D"))))))

        # update i_days and e_days (1. initialize to zero the newly turned I nodes; 2. add one to previously exposed or infectious nodes)
        old_I_nodes = list(set(i_nodes) - set(new_I_nodes))
        g_temp = g_temp.withColumn("i_days", when(g_temp.id.isin(new_I_nodes), 1).otherwise(when(g_temp.id.isin(old_I_nodes), g_temp.i_days+1).otherwise(0)))

        old_E_nodes = list(set(e_nodes) - set(new_E_nodes))
        g_temp = g_temp.withColumn("e_days", when(g_temp.id.isin(new_E_nodes), 1).otherwise(when(g_temp.id.isin(old_E_nodes), g_temp.e_days+1).otherwise(0)))
        
        # finish upating the vertices --> let's update the graph!
        g = GF.GraphFrame(g_temp, g.edges)

        duration += 1
        
        if len(i_nodes) == 0:
            print("TERMINATED: No more infectious nodes left to update")
            break
        
        nodes_counter.loc[step] = [len(s_nodes),len(e_nodes),len(i_nodes), len(r_nodes), len(h_nodes), len(d_nodes)]

    nodes_counter["duration"] = duration
    return nodes_counter, duration


#### ====================================
####  MAIN FUNCTIONS (interface with bash)
#### ====================================
## Argument parsers
parser=argparse.ArgumentParser(description="\n Monte Carlo simulation of disease transmission using different parameter values")

# Flags for probability parameters
probParam = parser.add_argument_group(title="Transition probabilities", description="Flags used to specify the transition probabilities from one state to another.")
probParam.add_argument('--p_is', default=0.5, type=float, help='User specified probability of infection. Default is 0.5.')
probParam.add_argument('--p_id', default=0.02, type=float, help='User specified transition probability from I to D for the UNHOSPITALIZED individuals. Default is 0.02.')
probParam.add_argument('--p_ih', default=0.06, type=float, help='User specified transition probability from I to H. Default is 0.06.')
probParam.add_argument('--p_ir', default=0.3, type=float, help='User specified transition probability from I to R. Default is 0.3.')
probParam.add_argument('--p_hr', default=0.15, type=float, help='User specified transition probability from H to R. Default is 0.15.')
probParam.add_argument('--p_hd', default=0.04, type=float, help='User specified transition probability from H to D. Default is 0.04.')

# Flags for duration (periods)
durationParam = parser.add_argument_group(title="Duration lengths", description="Flags used to specify the infectious period and the latent period.")
durationParam.add_argument('--t_latent', default=5, type=int, help='User specified duration of latent, i.e. how long an Exposed node becomes I node and can start infecting other people. Default is 5.')
durationParam.add_argument('--t_infectious', default=5, type=float, help='User specified duration of infectious, i.e. how long an I node is infectious for. After t_infectious days, an I node is no longer infectious. Default is 5.')

# Flags for initialization
initNodes = parser.add_argument_group(title="Initial state setup", description="Set the initial states of the nodes")
initNodes.add_argument('--num_i_seeds', default=100, type=int, help='Number of I nodes in initial stage. Default is 100.')
initNodes.add_argument('--num_time_steps', default=20, type=int, help='Number of maximum days to simulate. Default is 20.')

# Input file paths 
# important NOTE: the input should be the pre-processed network datasets - v_input requires "id"; e_input requires "src" and "dst"
ifile = parser.add_argument_group(title="Input Options", description="Input options to load network data.")
ifile.add_argument('--v_input', metavar='FILE_PATH', type=str, help='File path to the network data that contains the preprocessed vertices data.')
ifile.add_argument('--e_input', metavar='FILE_PATH', type=str, help='File path to the network data that contains the preprocessed edges data.')

# Output file paths
ofile = parser.add_argument_group(title="Output Options", description="Output directory and options to write to files.")
ofile.add_argument('--out', default='sim', metavar='FILE_PREFIX', type=str, help='File path to write the simulation results.')
ofile.add_argument("--make-full-path", default=False, action="store_true", help="option to make output path specified in --out if it does not exist. Default is False.")

########################################################

if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    logging.basicConfig(format='%(asctime)s %(message)s', filename=args.out + '.log', filemode='w', level=logging.INFO,datefmt='%Y/%m/%d %I:%M:%S %p')

    g = construct_graph(args)

    try:
        nodes_counter, duration = simulate(g, args.p_is, args.p_id, args.p_ih, args.p_ir, args.p_hr, args.p_hd, args.t_latent, args.t_infectious, args.num_i_seeds, args.num_time_steps)
        save_sim(args, nodes_counter, duration)

    except Exception:
        ex_type, ex, tb = sys.exc_info()
        logging.info( traceback.format_exc(ex) )
        raise

    finally:
        logging.info('Analysis finished at {T}'.format(T=time.ctime()) )
        time_elapsed = round(time.time()-start_time,2)
        logging.info('Simulation complete. Time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
