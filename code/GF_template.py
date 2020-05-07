#### INITIATE SPARK WITH THE FOLLOWING
# pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11

#### =================================
####  SETUP TO ENABLE GF AND PYSPARK
#### =================================

# graphframes
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 pyspark-shell")
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

# For nice printing
from IPython.display import display

####### SET UP Spark environment
spark = SparkSession.builder.appName('testtest').getOrCreate()

# Spark configuration
conf = (SparkConf().setMaster('local').setAppName('toy_graph'))

# Create an SQL context:
sql_context = SQLContext(sc)

# Spark context
# sc = SparkContext(conf=conf)

#### =================================
####  TO CALL AND RUN SERIAL CODES
#### =================================
# Option 1: if network_update_serial.py is in the current directory
# import network_update_serial as network_update

# Option 2: call direct path (NEED TO MODIFY THE FILE PATH)
api = '/home/ubuntu/CS205_FinalProject/testing'
script_path = api + '/network_update_serial.py'
os.system('cd {api};python {script_path} -config_files conf.ini'.format(script_path=script_path, api=api))

