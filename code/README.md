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

   A command line tool version of `network_updates_GF.py` that are callable from `bash`. Useful for Monte Carlo simulations. __All latest changes and functional improvements to network_update should be reflected in here.__ 

* `preprocess_network.py`:

   Pre-process the network datasets and format it as inputs to the Monte Carlo algorithm. *This script can also be used to modify our network data (e.g. enlarge the network size, add edges, etc.) to test assumptions.*
