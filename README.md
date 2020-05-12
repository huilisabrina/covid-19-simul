# covid-19-simul
### A graphical model and simulation of the spread of the coronavirus COVID-19

## Project Overview 
### Modeling COVID-19 with Big Data Methods
Given the state of the world at the moment, we hope to join the global effort to understand and combat the COVID-19 pandemic through this project. The main problem we are trying to solve is efficient simulation and graphical modeling of the spread of Covid-19 in a local community, using real Covid-19 incidents data and human interaction network data online. We want to apply our simulation model to both evaluate the impact of preventative measures on public health outcomes, as well as studying the progression of Covid-19 in a new demographic context (e.g. what if there were more people in the community, more interactions, etc).

We view this to be a big data problem, and we find that our application is data-intensive for two reasons. Firstly, the network datasets used in our project will likely have a large volume. The size of our dataset is a function of the number of individuals and connections, but also the number of time periods we want to model. And secondly, since disease transmission is probabilistic, there can be an infinite number of possible outcomes generated from the same initial graph. Due to the high degree of stochasticity in the simulations, we need to run a large number of simulations to ensure the robustness of our results. Big data parallel processing will allow us to conduct multiple simulations simultaneously, allowing us to generate histograms as well as point estimations of certain unknown parameters. 

### Modeling Approach and Existing Solutions
The details of the solution are provided below, but briefly speaking we use a two-layered approach to model subject-interaction in a community, and the spread of disease within those individuals/subjects. This combines two interesting modelling approaches: graphical models for modeling subject-interaction, and a standard SEIR model to subject this underlying graphical model to the epidemiological phenomenon of contagious disease-spread. 

Most of the existing work on this problem revolves around taking a mathematical/analytical approach, as opposed to the graphical approach we describe above. Specifically, the mathematical approach does not require an underlying graph structure, and instead uses differential equations to model how many individuals will migrate to and from each state and each time point 3. These differential equations are parametrized based on expert-understanding of the disease being modelled. Although this is a valid approach, the graphical approach can help capture the more nuanced nature of person-to-person interaction in the real-world. Furthermore, the graphical approach makes it very straightforward to incorporate stochasticity into the disease-transmission from one person to another. 

### Graphical Network and the SEIR Model
This is a two pronged-model, one for modelling subject-interaction in a community, and the other governing the spread of an epidemic through those interacting subjects: 

1. Graphical network for modelling subject-subject interaction in a community: We want to simulate an underlying network of nodes (people) and edges (interactions), so that any disease-spreading algorithm can decide who gets infected by any given currently- infected person. Although it is easy to generate random graphs for modelling such interactions (e.g. Erdos Renyi graphs), we quickly realized upon experimentation that the behavior of the epidemic can be very anomalous for such random graphs, since they connectivity-profile of random graphs do not approximate real-life interactions very well. In order to capture the nuanced - and sparse - nature of a real-life network, we resorted to using actual data from experimental studies. Specifically, we obtained a population network (list of edges) from an HIV survey done in the 1990s across eight different North American cities1. This led to eight disjoint sub-networks which can be modelled parallely, consisting of 49,355 nodes and 64,276 unique edges overall. In order to make the transition from an HIV network to a Covid-19 network, we made adjustments to the graph, such as adding random edges and nodes to mimic airborne transmission. 

2. SEIR model for epidemiological modelling of disease-spread: This is the epidemiological spreading process we overlaid onto our underlying population network. This is an extension of the standard SEIR (Susceptible - Exposed - Infected - Recovered) model which we augmented to include states for Death and Hospitalised. The model iterates through several time-steps (from time = 0 to time = model end). At each time step, individuals transition from one state to another based on some predetermined rules (shown as arrows in the picture below), and predetermined transition probabilities (indicated in the picture below). We calculated these transition probabilities based on literature reviews of the epidemiology of Covid-19 3. We keep track of individuals in each state at each time step. This allows us to calculate the size of the epidemic (in terms of cumulative number of people infected, hospitalized, deceased, etc), as well as the length (time from beginning of epidemic to time-step when there are no infected individuals left).

<img src="https://github.com/huilisabrina/covid-19-simul/blob/camille-dev/figures/Figs_ReadMe/SEIRmodel.png" width=600>

## Parallel Application and Programming Models

The main bottleneck in tackling a graph-based epidemic simulation is the large amount of complex graph data that must be manipulated and updated in order to fully model the spread of the epidemic. To address this issue, we use a Spark general-purpose big dataflow processing model, chosen over Map-Reduce due to its ability to handle complex data pipelines and in-memory data sharing across steps, as well as its general flexibility and simple deployment. Spark is run on a Hadoop cluster implemented through AWS EMR, and we use Pyspark to implement Spark in Python, our primary coding language. Within the Spark framework, our epidemic model is implemented using GraphFrames, a package for Apache Spark that manipulates DataFrame-based Graphs and includes various functionalities such as motif-finding, graph queries, and standard graph algorithms.[2] These functionalities are used to allow us to simulate the spread of the COVID-19 epidemic in a community, while taking advantage of the big data parallelism capabilities of Spark.

There are three levels of parallelism in our application. The first task-level parallelization occurs within our individual simulation pipeline. Our graph updating algorithm consists of a series of parallel transformations operated on the partitioned data. After the network data is pre-processed into two dataframes, with one holding the vertices and the other holding the edges, GraphFrames constructs the SEIRDH network graph from the dataframes, and the epidemic spread is simulated through a series of functions and transformations that are applied to the data. The Hadoop-supported Spark framework partitions the graph data into multiple segments that are distributed across multiple instances and threads in the cluster, and transformations on the segments of data are therefore able to be performed in parallel. Although the order in which the state changes occur is sequential and is not parallelized, within each state change step, parallelization occurs as described. 

The second level of parallelization occurs at the level of the connected components of the graph, referred to as “clusters” in the code. Given that the graph is composed of data from eight studies and essentially represents eight disjoint subgraphs with no overlapping vertices or edges, parallelizing at the subgraph/cluster level is an embarrassingly parallel problem. We implement this parallelization by splitting up the data into the eight separate datasets, and then running the network update simulation on these datasets in parallel using the same set of epidemic parameters. We then combine the results (i.e., the counts of nodes in each state and the duration of the epidemic) across clusters at the end in order to obtain the outcome of the network update simulation on the entire graph. 

The final level of parallelization is the coarsest-grained parallelization and occurs at the level of the Monte Carlo simulations. Each Monte Carlo simulation has a set of parameters and is completely independent of the other simulations. The parallelization of the Monte Carlo runs is performed in the same way as the cluster-level parallelization, with Spark running the simulations in parallel across nodes and cores. Our parallel execution model is SPMD, as we perform the same transformation on multiple data partitions. 

## Platform and Infrastructure

For prototyping and testing one single epidemic simulation on one subgraph, we use an Amazon EC2 m4.xlarge (8 vCore, 16 GiB memory) instance, running Spark locally. When introducing Monte Carlo simulations and a higher level of parallelism, we use Amazon Spark-configured EMR clusters constructed of m4.xlarge instances to exploit multi-core multi-node parallelization on the cloud. These clusters are easily scalable for additional worker nodes and allow for specific tuning for speed-up. Spark-configured EMR clusters are automatically constructed with the applications Hadoop 2.8.5 YARN, Ganglia 3.7.2, Spark 2.4.4, and Zeppelin 0.8.2. 

The GitHub repository that contains all code, data, and figures can be found at: https://github.com/huilisabrina/covid-19-simul. Source code and test cases can be found in the “code” folder, evaluation and input data sets are found in the “data” folder, and speed-up plots can be found in the “figures” folder.

## Code Design and Usage 

### Network Updater 
The core code, and its simplest implementation, can be found in **network_update_serial.py**. It depends on the python packages *random*, *numpy*, and *pandas*. First, functions are defined that dictate the flow along the SEIRHD model. These are *H_flow, I_flow, E_flow, S_flow*, which dictate the flow from hospitalized,infected, exposed, and susceptible states. Each of the flow functions takes in the relevant lists of nodes for each state, as well as a probability of flowing from the current state to each of the possible resulting states. When these functions are called, first it is tested that the probabilities given are sensible, then every node is probabilistically transitioned to a different possible state, or remains in the original state. After the flow function, the *Send_I* function ensures that an infected node will eventually be sent to state H, D, or R. Next, the *simulate* function is defined, which takes in pandas dataframes representing nodes and edges, state transition probabilities, timing functions to ensure that latent and infectious nodes propagate, and beginning numbers of nodes in states I, S, and H. This function calls the flow functions, taking steps until there are no more infected individuals. The number of nodes in each state for each timestep is recorded in the dataframe *node_counter*. The rest of the file consists of a toy example which uses a generated example network, in lieu of the data-based one that will be applied in more developed versions of this code. Parameters are given in the file for probabilities, latent and infections periods, and initial counts of nodes in states. The *simulate* function is called, and the file prints the *nodes_counter* dataframe as well as the number of time steps taken total. This file can be run using a typical command line directive to access a python file, such as “python network_update_serial.py”.

The next implementation of the code can be found in **network_update_GF.py**. This version also uses a toy implementation, but it has further dependencies and uses spark-supported graphframes rather than pandas dataframes. At the beginning of the file, the *future* package is used to ensure compatibility in python 2 and 3. In addition to numpy, random, and pandas, *reduce* from *functools*, *os*, *warnings*, *sys*, *re*, *logging*, *time*, *traceback*, and *argparse* are all called to interact with the system. The other two packages required are *graphframes* and *pyspark*, especially *pyspark.sql*. First, the spark configuration and setup are done, as well as the SQL context. Again, the flow functions are defined, using the same methods as in **network_update_serial.py**. Then, *simulate* is defined, taking the same inputs as it did before, except it takes a graphframe and not a dataframe. The dataframe *nodes_counter* is initialized. The graphframe *g* is set up with nodes and edges, and information about the neighbors, current state, and time spent in the I or E state is added to *g*. *g* is updated until there are no more infected individuals. The toy example is set up as it was in the previous code, except a graphframe is used instead of a dataframe. Again, the outputs are *nodes_counter* and the number of time steps taken. Using a spark-submit method, the file can be run from the terminal using a command such as “spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 network_update_GF.py”, ensuring that graphframes is included.

### Monte Carlo Implementation 
The next implementation of the code can be found in **network_update_GF_monte_carlo.py**. This version has some dependencies on other files, as outlined in the diagram below.

<img src="https://github.com/huilisabrina/covid-19-simul/blob/camille-dev/figures/Figs_ReadMe/carlo_flow.png" width=700>

The data input files necessary to run the code can be found in the "data" folder. **edge_list.csv** is the data provided from the HIV Transmission Network Metastudy Project, described earlier. It provides the social network used in the simulation, as a list of edges between nodes. The **params_input.csv** file provides parameters that will be selected during Monte Carlo simulations.

The file **preprocess_network.py** processes **edge_list.csv** for use as a network in the monte carlo simulation. It uses the packages, *random*, *numpy*, *pandas* and *itertools*. The csv is loaded into a dataframe and the vertices and edges are found, as dataframes. Then, a set of edges are chosen and the data is split into clusters, and the connectivity is adjusted to account for variances in edges for the same number of nodes, and adjustment that is ultimately used to model social distancing.

The bash script **monte_carlo_sim.bash** is used to call run the Monte Carlo simulation. First, **preprocess_network.py** is called on **edge_list.csv**. Next, the parameters are parsed in from **params_input.csv**. Then, **network_update_GF_monte_carlo.py** is called using spark-submit, with a number of objects passed in, including the graphframes package, the vertices and edges found in preprocessing, and parameters. The simulations are counted. 

**network_update_GF_monte_carlo.py** has the same dependencies as **network_update_GF.py**, and the flow and simulation are done in the same way. However, instead of using a toy example. it uses the preprocessed edge list. After these functions, the argument parser is stated, followed by the flags for probability, latent period, initial counts per state, and input and output file paths, which are supplied by the bash script. In the main function, the simulation is called and timed, with the aid of some helper functions. The output files are **sim.txt** and **sim.log**. **sim.txt** gives the number of nodes in each state for every time step, as shown below.

```
n_s	n_e	n_i	n_r	n_h	n_d	duration
576	0	20	0	0	0	20
568	8	12	8	0	0	20
566	10	6	13	0	1	20
564	12	3	16	0	1	20
564	12	1	17	1	1	20
564	12	1	17	1	1	20
562	14	8	18	1	1	20
561	15	4	23	2	1	20
559	17	3	25	2	2	20
557	19	2	26	2	2	20
557	19	1	27	2	2	20
555	21	3	27	2	2	20
555	21	2	31	0	2	20
554	22	3	32	0	2	20
553	23	2	35	0	2	20
553	23	1	36	0	2	20
553	23	2	37	0	2	20
553	23	1	38	0	2	20
553	23	2	38	0	2	20
552	24	3	38	0	2	20
```

**sim.log** gives information such as the steps taken, parameters, and time elapsed, as shown below.

```
2020/05/11 05:40:48 AM Setting up graph data with 596 nodes, 132982 edges and 1 clusters
2020/05/11 06:18:49 AM 
 Summary of simulation: 
------------------------------------- 
Duration: 20
------------------------------------- 
p_is: 0.5
------------------------------------- 
p_id: 0.0134
------------------------------------- 
p_ih: 0.0678
------------------------------------- 
p_ir: 0.3945
------------------------------------- 
p_hr: 0.19725
------------------------------------- 
p_hd: 0.0419
------------------------------------- 
t_latent: 5
------------------------------------- 
t_infectious: 5.0

2020/05/11 06:18:49 AM Analysis finished at Mon May 11 06:18:49 2020
2020/05/11 06:18:49 AM Simulation complete. Time elapsed: 38.0m:
```


There are also a few other bash scripts in the code file, for specialized usage in the place of **monte_carlo_sim.bash**. These scripts still require network preprocessing, but **monte_carlo_sim.bash** references the **params_input.csv**, while these scripts do not, and set default values are used. 

**diff_graphs.bash** runs the Monte Carlo network updater with clusters for different levels of connectivity, as expressed by a number of edges relative to the number of nodes. Three different networks are run with edge count adjustments, using the same disease propagation parameters. 

**local_run.bash** runs the Monte Carlo network updater for a single execution, using default values rather than reading them from **params_input.csv** It uses a single worker node. **cluster_run.bash** is the same as **local_run.bash**, but it does use multiple worker nodes and threads.

### Data-Partitioned (Cluster) Monte Carlo Implementation

To run with parallelized clusters, additions have to be made to the existing workflow, as shown below.

<img src="https://github.com/huilisabrina/covid-19-simul/blob/camille-dev/figures/Figs_ReadMe/cluster_flow.png" width=700>

The cluster-parallelized, final version of the main code is **network_update_GF_monte_carlo_cluster.py** . This file is just like **network_update_GF_monte_carlo_cluster.py**, except it has a different spark configuration. 

The bash script **parallel_clusters.bash**, calls the network update function and then aggregates the clusters by calling on **combine_clusters.py**, which serves as a final step to combine data from across clusters that have been run in parallel. In this case, sim.txt is not the final output. It depends on *future*, *random*, *numpy*, *pandas* and *argparse*. *argparse* is used to process the flags for the number of clusters, input, and output. The final node counts are aggregated in a dataframe and the results are saved. 

All tests were performed on AWS m4.xlarge instances. Here are the software versions for reproducibility: 
* GraphFrames: version 0.6.0
* Spark: version 2.3-s 2.11
* Python version: 2.7.17
* AWS m4.xlarge instances
* EMR mode: 1 master + 6 workers

## Performance Evaluation

### Speed-up and Scaling

Performance can be evaluated through examining runtimes across different numbers of cores and threads per core. Strong scaling with a fixed problem size and an increasing number of processors is an appropriate way to evaluate performance because the underlying social network, and therefore the total number of vertices and edges in the graph, is fixed. 

The following plot provides a comparison of the runtimes obtained after increasing the number of cores (where each core is an m4.xlarge instance) and the number of threads per core. We see that having 5 cores seems to provide the most speed-up, for both the single-thread and double-thread scenarios. Focusing our attention on the single-thread runtimes, as the number of cores increases, the runtime steadily decreases up until 5 cores are used, but then increases dramatically once 6 cores are used. This is likely due to the substantial parallelization overheads required to use 6 cores. In contrast, the runtimes when two threads per core are used are generally shorter than the single-thread runtimes, but show very little improvement, if any, when the number of cores is increased. This is also likely due to the trade-off between parallelization overheads and parallelization speed-up, as more communication is required when multiple threads are used.

<img src="https://github.com/huilisabrina/covid-19-simul/blob/master/figures/speed_up_cluster.png" width=600>

When comparing one thread per core to two threads per core, the highest speed-up can be seen when only two cores are used. In this case, speed-up is around 1.4. As the number of cores increases, the speed-up obtained by using multiple threads quickly decreases and in some circumstances disappears completely. When evaluating the effect of an increasing number of processors compared to a baseline of two processors, for one thread per processor, speed-up increases quickly up until 5 processors, then drops; for two threads per processor, no apparent speed-up trend can be observed. These results indicate that when a limited number cores are available, it is optimal to use a double-threaded approach; however, when there are 5 processors available, the single-threaded approach performs just as well, if not better, than the double-threaded approach. One thing to note here is the inherent stochasticity built into the graph updating step may lead to different runtimes depending on the run, so these results may differ slightly if we were to repeat the experiment.

### Overheads and Optimization

Some reasons why parallelization does not lead to substantial speed-up include inherent serialization in the code, such as the order in which disease states must be updated and the dependence of subsequent time steps on the results of previous time steps. These serial portions result in bottlenecks for parallelization and limit the possible speed-up, as described by Amdahl’s Law. In addition, parallelization overheads accrue as more cores are used and, consequently, result in large runtime costs that offset the benefits of parallelization. These overheads arise due to communication, synchronization, load balancing, etc., between processors. Given that we use Spark GraphFrames as our main vehicle for parallelization, we can perform some tuning to find the optimal number of threads and cores to use, but we are unable to perform fine-tuned load-balancing to minimize waiting-times. A modification of the project could include using MPI to parallelize across study clusters or Monte Carlo simulations, and to use load-balancing techniques in MPI. 

## Challenges and Advanced Features

The main package used in this project was Spark-supported GraphFrames.[2] Graph frames are created from vertex and edge DataFrames. Once created, they provide simple graph queries, such as node degree, as well as motif finding to search for structural patterns in a graph. Given that the Graph query function in GraphFrames does not have a built-in neighbor-finding function, we used the motif-finding capabilities to produce a work-around way to find all the neighbors of a given vertex during each update step. 

```
    neighbor = g.find("(a)-[e]->(b)").drop("e").groupBy('a.id').agg(collect_list('b.id').alias('neighbors'))
    g_neighbor = neighbor.join(g.vertices, ['id'], "right_outer")
    g = GF.GraphFrame(g_neighbor, g.edges)
```

Another challenge with using GraphFrames is the immutability of the data structure. Because we cannot apply updates to the graph directly, we have to generate a new graph each time we choose to make modifications. To avoid prohibitive overheads, we chose to update the graph per time step rather than for each state change of each node, using lists in each of the state change functions instead. This resulted in substantial overhead, but the overhead was much smaller than what would have occurred had we updated the graph every node state change. 

To simulate the effect of public health interventions, we ran our model using different characterizations of disease transmission and network structure. For example, in one Monte Carlo iteration (shown on the left, below), we increased the infectious and latent periods to simulate situations where testing might be delayed or lacking and found that a longer infectious period increased the number of total nodes that were infected, although a longer latent period (without infectivity) offset this increase. However, considering current evidence on COVID-19 suggests individuals can be infectious while latent, the result pertaining to longer latent periods is less directly applicable. We also varied the degree of connectivity of our graph (shown on the right, below) by increasing or decreasing the number of edges to examine the effect of social distancing, or lack thereof. Interestingly, we did not see much change, but the results of a different run could differ due to the inherent stochasticity in our model. 

<img src="https://github.com/huilisabrina/covid-19-simul/blob/camille-dev/figures/Figs_ReadMe/finalfigs.png" width=700>

One interesting aspect of our results is the epidemic dies out quickly and very few nodes are infected. After much debugging, we discovered the source of the issue to be our implementation of the SEIRHD model. This was another one of our main challenges. Given the lack of existing implementation of epidemic network models in GraphFrames, we had to produce both serial and parallel versions of this epidemic model. Our hope was to include parameters that best reflect current knowledge on COVID-19, such as the latent and infectious periods. We adapted our model by utilizing the vertex attribute functionality in GraphFrames to keep track of the state of the nodes as well as the time elapsed since exposure or infection. Unfortunately, this adaptation led to some issues with the flow of the state changes that we only discovered after using parallelization to scale up our simulations; however, future versions of our code could easily modify the graph updating functionality to produce more accurate epidemic estimates.

Another challenge encountered during implementation was the sheer size of the network data. When calculating speed-up, we were unable to obtain runtime information for the single-core, single-thread scenario due to memory and time constraints. In addition, every run of the epidemic model took substantial time, so we had to use various “toy” or subsetted data sets when creating and debugging code, and we were only able to run a limited number of scenarios for relatively short epidemic durations.

## Goals and Future Work

### Goals and Insights 
When we began this project, we hoped to apply the skills learned in this class to contribute, in some small way, to the global effort to understand and combat the COVID-19 pandemic. It was our hope that a graph-based approach to simulating the pandemic would be able to incorporate a realistic underlying network structure upon which the extent and effect of a pandemic could be simulated. We also brainstormed ways to make the network reflect the unique person-to-person transmission patterns of COVID-19, such as randomly adding edges to the graph to simulate airborne transmission. To further increase real-life approximation, we estimated our epidemic transmission parameters using peer-reviewed literature, as well as the EpiEstim R package software, which uses a Bayesian approach to quantify transmissibility over time in an epidemic.[4,5] Our goal was to use big data parallelization techniques to create efficient simulations to model a community-level COVID-19 epidemic in a way that was realistic, flexible, and could be used to examine the impact of public health interventions, such as social distancing, as well as the progression of the epidemic in developing nations, where healthcare infrastructure may be less robust. We believed that the speed-up gains from parallelizing the code would enable graphical simulations on a much larger scale and would therefore provide more interesting and directly applicable results for public health and public policy. 

In terms of achieving our goals, we were able to complete a fully functional epidemic model implemented using GraphFrames. We were able to parallelize our results by running Spark in local mode, and then Spark in cluster mode with multiple cores and threads. As a result, we were able to gain some speed-up through parallelization, with the fastest runtime occurring with 5 single-thread cores, but substantial overheads inhibited further speed-up when we increased the number of cores and threads. Another goal that was achieved was evaluating the progression of the epidemic for sparser or denser network graphs, meant to approximate the effect of social distancing measures or the lack thereof. We were also able to run the simulations over a range of parameter values, varying the transmission probability and the latent and infectious periods, in order to model the spread of the epidemic under different scenarios. These reflect the uncertainty in the parameter estimates and provide a way to analyze which values of the parameters seem most accurate, with accuracy perhaps being determined by comparing the results to current COVID-19 datasets.

### Areas of Improvement

A key improvement to our results would be to modify this SEIRHD model implementation, particularly the interplay between the probability of transitioning from I to R and the incorporation of the infectious period in the state updates. This modification would greatly improve the accuracy of our simulations. Other areas of improvement include working to incorporate the entire underlying network determined by our HIV study data set, as well as increasing the allowed duration of the epidemic. We were unable to run our model on the entire network due to runtime and credit constraints. Even with our parallelization through Spark, our simulation took a substantial amount of time to run. Due to similar runtime constraints, we were only able to run the simulations with a very limited number of time steps. Future work to increase parallelization and speed-up would bring us closer to the initial goal of running the model on the entire data set and obtaining long-term epidemic forecasts. Possible avenues for achieving this increased parallelization include using the cluster-level parallelization code and pipeline established in the repository, and/or incorporating MPI and MapReduce.

### Future Work

Our project can be extended by future work in numerous ways. One idea would be to weight the various edges in the network by the strength of the tie (sexual versus social) and allow the probability of transmission to differ depending on the strength of the relationship. Other attributes of the vertices, such as age and gender, could also be incorporated to allow the transmission probabilities as well as the recovery, hospitalization, and death probabilities to differ depending on these attributes, thereby providing a more realistic representation of the pandemic spread. An additional extension would be to incorporate more flexibility into the transmission parameters. For example, one could allow them to be time-dependent, or draw them from a probability distribution and quantify the range and likelihood of possible epidemic outcomes. Finally, our graphical network has the potential to produce information on additional aspects of an epidemic, such as the burden on the healthcare system and the mortality rate.

## Citations
1. HIV Transmission Network Metastudy Project: An Archive of Data From Eight Network Studies, 1988--2001 (ICPSR 22140)
2. Ankur Dave, Alekh Jindal, Li Erran Li, Reynold Xin, Joseph Gonzalez, and Matei Zaharia. 2016. GraphFrames: an integrated API for mixing graph and relational queries. In Proceedings of the Fourth International Workshop on Graph Data Management Experiences and Systems (GRADES ’16). Association for Computing Machinery, New York, NY, USA, Article 2, 1–8. DOI:https://doi.org/10.1145/2960414.2960416 
3. Pan A, Liu L, Wang C, et al. Association of Public Health Interventions With the Epidemiology of the COVID-19 Outbreak in Wuhan, China. JAMA. Published online April 10, 2020. doi:10.1001/jama.2020.6130
4. Cori et al. 2013 "A New Framework and Software to Estimate Time-Varying Reproduction Numbers During Epidemics"
5. Wang, C., Liu, L., Hao, X., Guo, H., Wang, Q., Huang, J., ... & Wei, S. (2020). Evolving epidemiology and impact of non-pharmaceutical interventions on the outbreak of coronavirus disease 2019 in Wuhan, China. medRxiv.
