#### INITIATE SPARK WITH THE FOLLOWING
# pyspark --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11
#### to enable lib packages
# pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11

#### =================================
####  SETUP TO ENABLE GF AND PYSPARK
#### =================================

# graphframes
from graphframes import *
from graphframes import graphframe as GF
from graphframes.lib import AggregateMessages as AM
from graphframes.examples import Graphs

# MPI
from mpi4py import MPI

# Basic Spark classes
from pyspark import SparkContext, SparkConf

# SQL + Spark
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.functions import col, lit, udf, when, concat
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Basic packages
import random
import numpy as np
import pandas as pd
from functools import reduce
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark-shell")
# import scipy.stats as ss
# import matplotlib.pyplot as plt
# %matplotlib notebook


# For nice printing
from IPython.display import display

####### SET UP Spark environment
# spark = SparkSession.builder.appName('testtest').getOrCreate()

# Spark configuration
conf = (SparkConf().setMaster('local').setAppName('toy_graph'))
# Spark context
sc = SparkContext(conf=conf)
# Create an SQL context:
sql_context = SQLContext(sc)




#### =================================
####  GRAPH UPDATING FUNCTIONS
#### =================================

#Carry out the H -> R,D process for one time step
def H_flow(h_nodes, r_nodes, d_nodes, p_hr, p_hd):

    #Input check
    assert 1-p_hr-p_hd > 0, "Unrealistic p_hr and p_hd input."

    new_recoveries = []
    new_deaths = []

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
    """
    TO DO: allow one node in Infected to affect multiple nodes in S.

    Currently, one I node can only affect one S.
    """
    new_E_nodes = []
    for node in i_nodes:
        # extract i_days from the graph
        i_days = g.vertices.filter(col("id").isin([node])).select("i_days").toPandas()["i_days"][0]

        # an I node can only continue to infect people if t_infectious has not been reached
        if i_days <= t_infectious:

            #Select neighbors (only infect via existing edges)
            neighbors = g.vertices.filter(col("id").isin([node])).select("neighbors").toPandas()["neighbors"][0]
            
            if len(neighbors) != 0:
                infected = random.choice(neighbors)
                if infected in s_nodes: #One infection per person per period
                    flip = np.random.binomial(1, p_is)
                    if flip == 1: #Infect others
                        new_E_nodes.append(infected)
                        e_nodes.append(infected)
                        s_nodes.remove(infected)
            else: #Assuming network stays constant, a I node w/o neighbors flow to R,D,H
                i_nodes.remove(node)
                send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)

        else: # if I node is no longer infectious (i.e. i_days > t_infectious), go to one of R,H,D
            i_nodes.remove(node)
            send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)

    return new_E_nodes

#Send I node to one of D,H,R once infectious period is reached
def send_I(i_node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir):

    #Reweight p_id, p_ih, p_ir so that transition is definitive
    p_id, p_ih, p_ir = [x/sum([p_id, p_ih, p_ir]) for x in [p_id, p_ih, p_ir]]
    assert p_ir == 1-p_id-p_ih, "Wrong reweighting of p_id, p_ih and p_ir!"

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
             num_i_seeds, num_s_seeds, num_h_seeds, num_time_steps):

    # select the vertices as a list
    nodes = list(g.vertices.select("id").toPandas()["id"])
    
    i_nodes = list(random.sample(nodes, num_i_seeds))
    s_nodes = list(random.sample(set(nodes) - set(i_nodes), num_s_seeds))
    h_nodes = list(random.sample(set(nodes) - set(i_nodes) - set(s_nodes), num_h_seeds)) #Need initial values for first round of updating
    e_nodes = []
    r_nodes = []
    d_nodes = []
    duration = 0

    nodes_counter = pd.DataFrame(columns=["n_{}".format(x) for x in ['s','e','i','r','h','d']], index=[i for i in range(num_time_steps)])
    nodes_counter.loc[0] = [len(s_nodes),len(e_nodes),len(i_nodes), len(r_nodes), len(h_nodes), len(d_nodes)]

    # ADD "neighbors" column: select neighbors of a node based on a graph (used in S_flow step)
    neighbor = g.find("(a)-[e]->(b)").drop("e").groupBy('a.id').agg(collect_list('b.id').alias('neighbors'))
    g_neighbor = neighbor.join(g.vertices, ['id'], "right_outer")
    g = GF.GraphFrame(g_neighbor, e)

    # ADD "state" column
    # At t0: number of I nodes and H nodes are based on user-defined functions. ALL OTHER NODES are assumed to be S
    g_temp = g.vertices.withColumn("state",when(g.vertices.id.isin(i_nodes), "I").otherwise(when(g.vertices.id.isin(h_nodes), "H").otherwise("S")))
    
    # ADD "i_days" and "e_days" column 
    # At t0: 1 for all I_nodes and 0 for all others
    g_temp = g_temp.withColumn("e_days", lit(0))
    g_temp = g_temp.withColumn("i_days", when(g.vertices.id.isin(i_nodes), lit(1)).otherwise(lit(0)))
    g = GF.GraphFrame(g_temp, e)

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
        g = GF.GraphFrame(g_temp, e)

        duration += 1
        
        if len(i_nodes) == 0:
            print("TERMINATED: No more infectious nodes left to update")
            break
        
        nodes_counter.loc[step] = [len(s_nodes),len(e_nodes),len(i_nodes), len(r_nodes), len(h_nodes), len(d_nodes)] 
    
    return nodes_counter, duration

#### =================================
####  TESTING TOY EXAMPLE
#### =================================

#The following part with be substituted with data import and reformatting

# Start with dataframes in pandas
sim_pop_size = 100
sim_edge_count = 200
sim_n_cluster = 3
print("Setting up graph data with {} nodes, {} edges and {} clusters".format(sim_pop_size, sim_edge_count, sim_n_cluster))

v = pd.DataFrame({'id' : [i for i in range(sim_pop_size)],
    'age' : list(np.random.randint(100, size=sim_pop_size)),
    'sex' : list(np.random.randint(2, size=sim_pop_size)),
    'cluster' : list(np.random.randint(8, size=sim_pop_size))
})

e = pd.DataFrame(columns=["src","dst"], index=[i for i in range(sim_edge_count)])
for i in range(sim_edge_count):
    e.loc[i, "src"] = random.sample(list(v["id"]), 1)[0]
    e.loc[i, "dst"] = random.sample(list(set(v["id"])-set([e.loc[i, "src"]])), 1)[0]

# Coin dataframes into SQL for graphframes
v_schema = StructType([StructField("id", IntegerType(), True), 
                     StructField("age", IntegerType(), True),
                     StructField("sex", IntegerType(), True),
                     StructField("cluster", IntegerType(), True)])
v = sql_context.createDataFrame(v, schema = v_schema).dropDuplicates(['id'])

e_schema = StructType([StructField("src", IntegerType(), True), 
                     StructField("dst", IntegerType(), True)])
e = sql_context.createDataFrame(e, schema = e_schema)

# Generate graph before simulation starts
g = GF.GraphFrame(v, e)

# Define parameters
p_is = 0.5
p_id = 0.0419
p_ih = 0.0678
p_ir = 0.0945*2
p_hr = 0.03945
p_hd = 0.0419

t_latent = 7
t_infectious = 5

num_i_seeds = 10
num_s_seeds = 30
num_h_seeds = 2
num_time_steps = 5

# Run simulations
nodes_counter, duration = simulate(g, p_is, p_id, p_ih, p_ir, p_hr, p_hd, t_latent, t_infectious, num_i_seeds, num_s_seeds, num_h_seeds, num_time_steps)
print("Evolvement of nodes in each category")
print(nodes_counter)
print("Duration: {}".format(duration))
