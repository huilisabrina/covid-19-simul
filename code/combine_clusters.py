#---------------------------------------------- 
# CS 205 Final Project
# Cluster combining functions for final output
#
# To be called by monte_carlo_sim.bash
#----------------------------------------------

## Argument parsers
parser=argparse.ArgumentParser(description="\n Combination of results across study clusters")

# Flags for clusters
clusterParam = parser.add_argument_group(title="Cluster properties", description="Flags used to specify the cluster properties of the network graph.")
clusterParam.add_argument('--n_clusters', default=8, type=int, help='Number of disjoint clusters (connected components) in the network graph')

# Input file paths 
ifile = parser.add_argument_group(title="Input Options", description="Input options to load network data.")
ifile.add_argument('--input', metavar='FILE_PATH', type=str, help='File path to the network data that contains the simulation output.')

# Output file paths
ofile = parser.add_argument_group(title="Output Options", description="Output directory and options to write to files.")
ofile.add_argument('--out', default='sim', metavar='FILE_PREFIX', type=str, help='File path to write the simulation results.')
ofile.add_argument("--make-full-path", default=False, action="store_true", help="option to make output path specified in --out if it does not exist. Default is False.")

if __name__ == '__main__':
    args = parser.parse_args()

    # Load pre-processed datasets
    n_clusters = args.n_clusters
    pd.read_csv(args.v_input, index_col=False, delim_whitespace=True)
    v = pd.read_csv(args.v_input, index_col=False, delim_whitespace=True)
    e = pd.read_csv(args.e_input, index_col=False, delim_whitespace=True)



    logging.info("Setting up graph data with {} nodes, {} edges and {} clusters".format(v.shape[0], e.shape[0], len(v["cluster"].unique())))
