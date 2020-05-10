#------------------------------------------------------
# CS 205 Final Project
# Combine results across clusters for full graph output
#
# To be called by monte_carlo_sim_clusters.bash
#------------------------------------------------------

## Argument parsers
parser=argparse.ArgumentParser(description="\n Combination of results across study clusters")

# Flags for clusters
clusterParam = parser.add_argument_group(title="Cluster properties", description="Flags used to specify the cluster properties of the network graph.")
clusterParam.add_argument('--names', default=8, type=int, help='Names of the disjoint clusters (connected components) in the network graph')

# Input file paths 
ifile = parser.add_argument_group(title="Input Options", description="Input options to load network data.")
ifile.add_argument('--input', metavar='FILE_PATH', type=str, help='File path to the network data that contains the simulation output.')

# Output file paths
ofile = parser.add_argument_group(title="Output Options", description="Output directory and options to write to files.")
ofile.add_argument('--out', default='sim', metavar='FILE_PREFIX', type=str, help='File path to write the simulation results.')
ofile.add_argument("--make-full-path", default=False, action="store_true", help="option to make output path specified in --out if it does not exist. Default is False.")

#######################################################

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize full graph dataframe
    agg_df = pd.DataFrame(columns=["n_{}".format(x) for x in ['s','e','i','r','h','d']], index=[i for i in range(num_time_steps)])
    agg_df["duration"] = 0

    # For each cluster, load output and add results element-wise to the full graph dataframe
    duration = 0
    for name in args.names

        temp_df = pd.read_csv(args.input + "_" + name +".txt", index_col=False, delim_whitespace=True)
        temp_dur = temp_df.at[0, 'duration']
        if temp_dur > duration:
            duration = temp_dur
        agg_df.add(temp_df, fill_value = 0)

    # Set duration to be the longest epidemic duration across clusters
    agg_df["duration"] = duration

    # Save results in csv file
    agg_df.to_csv(args.out+".txt", sep='\t', index=False, na_rep="NA")
