import random
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
%matplotlib notebook

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
def E_flow(e_nodes, i_nodes, t_latent):
    """
    TO DO: add an attribute to the nodes, node.days_elapsed()
    to keep track of the days since a node becomes E.

    Currently, t_latent is disabled and effectively 1
    i.e. E becomes I in the next period
    """

    for node in e_nodes:
        i_nodes.append(node)
#         if node.days_elapsed() == t_latent:
            # i_nodes.append(node)
#         else:
#             node.days_elapsed() =+ 1
    return 0

#Carry out the I --> S --> E spreading process for one time step.
def S_flow(df_edges, s_nodes, i_nodes, e_nodes, p_is, t_infectious):
    """
    TO DO: 
    1) add an attribute to the nodes, node.days_elapsed()
    to keep track of the days since a node becomes I.

    Currently, t_infectious is disabled and effectively Inf
    i.e. once infectious, always infectious.

    Goal: I is only infectious for t_infectious periods. 
    Once no longer infectious, forced to be sent to R,D or H.

    2) allow one node in Infected to affect multiple nodes in S.

    Currently, one I node can only affect one S.
    """
    new_infect = []
    for node in i_nodes:
        # if node.days_elapsed() <= t_infectious:

            #Select neighbors (only infect via existing edges)
            neighbors = list(df_edges.loc[df_edges['src']==node, "dst"]) + list(df_edges.loc[df_edges['dst']==node, "src"])
            
            if len(neighbors) != 0:
                infected = random.choice(neighbors)
                if infected in s_nodes: #One infection per person per period
                    flip = np.random.binomial(1, p_is)
                    if flip == 1: #Infect others
                        new_infect.append(infected)
                        e_nodes.append(infected)
                        s_nodes.remove(infected)
                        # infected.days_elapsed() = 0
            else: #Assuming network stays constant, a I node w/o neighbors flow to R,D,H
                i_nodes.remove(node)
                send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)

        #     node.days_elapsed() =+ 1
        # else:
        #     i_nodes.remove(node)
        #     send_I(node, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)

    return 0

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
def simulate(df_nodes, df_edges, 
             p_is, p_id, p_ih, p_ir, p_hr, p_hd, 
             t_latent, t_infectious, 
             num_i_seeds, num_s_seeds, num_h_seeds, num_time_steps):
    nodes = list(df_nodes['id'])
    i_nodes = list(random.sample(nodes, num_i_seeds))
    s_nodes = list(random.sample(set(nodes) - set(i_nodes), num_s_seeds))
    h_nodes = list(random.sample(set(nodes) - set(i_nodes) - set(s_nodes), num_h_seeds)) #Need initial values for first round of updating
    e_nodes = []
    r_nodes = []
    d_nodes = []
    duration = 0

    nodes_counter = pd.DataFrame(columns=["n_{}".format(x) for x in ['s','e','i','r','h','d']], index=[i for i in range(num_time_steps)])
    nodes_counter.loc[0] = [len(s_nodes),len(e_nodes),len(i_nodes), len(r_nodes), len(h_nodes), len(d_nodes)]

    # TO DO: allow p_is to be a vector of rates
    for step in range(1, num_time_steps+1):
        H_flow(h_nodes, r_nodes, d_nodes, p_hr, p_hd)
        I_flow(i_nodes, r_nodes, h_nodes, d_nodes, p_id, p_ih, p_ir)
        E_flow(e_nodes, i_nodes, t_latent)
        S_flow(df_edges, s_nodes, i_nodes, e_nodes, p_is, t_infectious)
                   
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
#Format edges and vertices
sim_pop_size = 500
sim_edge_count = 5000
sim_n_cluster = 8
print("Setting up graph data with {} nodes, {} edges and {} clusters".format(sim_pop_size, sim_edge_count, sim_n_cluster))

v = pd.DataFrame({'id' : [i for i in range(pop_size)],
    'age' : list(np.random.randint(100, size=pop_size)),
    'sex' : list(np.random.randint(2, size=pop_size))
})

e = pd.DataFrame(columns=["src","dst","cluster"], index=[i for i in range(sim_edge_count)])
for i in range(sim_pop_size):
    e.loc[i] = pd.Series({"src": random.sample(list(v["id"]), 1)[0], "dst": random.sample(list(v["id"]), 1)[0], "cluster": random.sample([i for i in range(sim_n_cluster)], 1)})


#Define parameters
p_is = 0.5
p_id = 0.0419
p_ih = 0.0678
p_ir = 0.3945*2
p_hr = 0.3945
p_hd = 0.0419*10

t_latent = 7
t_infectious = 5

num_i_seeds = 100
num_s_seeds = 100
num_h_seeds = 20
num_time_steps = 20

#Run simulations
nodes_counter, duration = simulate(v, e, p_is, p_id, p_ih, p_ir, p_hr, p_hd, t_latent, t_infectious, num_i_seeds, num_s_seeds, num_h_seeds, num_time_steps)
print("Evolvement of nodes in each category")
print(nodes_counter)
print("Duration: {}".format(duration))
