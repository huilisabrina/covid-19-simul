# CS 205 Final Project Script Documentations

* `GF_template.py`: 
   
   Sample scripts for setting up GraphFrames in python. 

   Contains useful packages to import for graph queries, working with `pyspark.sql.dataframe` objects, etc. 
   
   For more details on how to run GraphFrames with Spark on a AWS instance, see Issue #2.

* `network_update_serial.py`:

   Network updating functions that only use `pandas`. `numpy` and `random`. 
   
   The simplest implementation of our network updating functions. 
   
   Contains toy example which is easy and fast to run. 

* `network_update_GF.py`:

   A more developed network updating algorithm that incorporates GraphFrames and `pyspark.sql.dataframe` objects. 
   
   Latent period, infection period, are enabled. Graphs (states of the nodes) are updated per time step.

* `network_update_GF_monte_carlo.py`:

   A command line tool version of `network_updates_GF.py` that are callable from `bash`. Useful for Monte Carlo simulations. By default uses two threads of local mode. To be called by all bash scripts.

* `network_update_GF_monte_carlo_cluster.py`:

   A dual version of `network_update_GF_monte_carlo.py` that is used on EMR cluster. To be called by `monte_carlo_sim.bash` and `parallel_clusters.bash`. 

* `monte_carlo_sim.bash`:

   Master script that calls the network update functions, using an Excel file that specifies parameter values for each run. 

* `parallel_clusters.bash`:

   Master script for cluster parallelization. Calls network updates functions, then `combine_clusters.py` to consolidate the cluster-specific results.

* `diff_graphs.bash`:

   Master script to test hypothesis: How does high/medium/low level of graph connectivity of the graph affect the epidemic outcomes?
   
* `preprocess_network.py`:

   Pre-process the network datasets and format it as inputs to the Monte Carlo algorithm. Split up the input datasets by clusters, for cluster-level parallelization. 

   Generate network data with different degrees of connectivity, for assumptions tests.

* `combine_clusters.py`:
   
   Python script to combine the cluster-level outputs. To be called by `parallel_clusters.bash`.
