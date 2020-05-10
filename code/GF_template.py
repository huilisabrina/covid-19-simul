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
conf = SparkConf().setMaster('local[4]').setAppName('template')
# Spark context
sc = SparkContext(conf=conf)
# surpress logging
sc.setLogLevel("ERROR")
# Create an SQL context:
sql_context = SQLContext(sc)

#### =================================
####  TO CALL AND RUN SERIAL CODES
#### =================================
# Option 1: if network_update_serial.py is in the current directory
# import network_update_serial as network_update

# Option 2: call direct path (NEED TO MODIFY THE FILE PATH)
api = '/home/ubuntu/CS205_FinalProject/testing'
script_path = api + '/network_update_serial.py'
os.system('cd {api};python {script_path} -config_files conf.ini'.format(script_path=script_path, api=api))

