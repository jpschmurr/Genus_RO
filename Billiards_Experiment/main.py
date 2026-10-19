from experiment_functions import billiard_graph_min_genus_experiment, billiard_graph_min_genus_experiment_n_runs, billiard_bracket_type_one_experiment, billiard_bracket_type_two_experiment, billiard_bracket_type_three_experiment, billiard_bracket_type_four_experiment, billiard_bracket_type_five_experiment
from rotation_and_plotting_functions import neighbor_easy_swap, neighbor_segment_reverse, neighbor_edge_dart, neighbor_full_shuffle, propose_multi_neighbor 
'''
Type One: [1,1,1,2k-3] 
Type Two: [1,2,3,2k]
Type Three: [1,2,3,4,3k+2] 
Type Four: [k,k+1,k+2,k+3]
Type Five: [1,1,1,1,3k-4]
'''

if __name__ == "__main__":

    print("=== Billiard Graph Minimum Genus Experiment ===")

    # --------------------------------------------------------
    # Enter the billiard bracket.
    # --------------------------------------------------------

    bracket_input = input(
        "Enter billiard bracket "
        "(recommended: 1 2 3 2): "
    )

    bracket = [
        int(x)
        for x in bracket_input.split()
    ]

    neighbors = [
        ("Easy: Single-vertex 2-swap", neighbor_easy_swap),
        ("Medium: Segment reversal", neighbor_segment_reverse),
        ("Medium: Edge-targeted dart", neighbor_edge_dart),
        #("Hard: full-vertex shuffle", neighbor_full_shuffle),
        #("Multi-neighborhood (weighted)", propose_multi_neighbor)
    ]

    #billiard_graph_min_genus_experiment(bracket)
    billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=3, max_iterations=2000000, init_temperature=3.0, final_temperature=1e-5, funcs=neighbors)
    #billiard_bracket_type_one_experiment(n_runs=3, m=20)
    #billiard_bracket_type_two_experiment(n_runs=3, m=20)
    #billiard_bracket_type_three_experiment(n_runs=3, m=20)
    #billiard_bracket_type_four_experiment(n_runs=3, m=20)
    #billiard_bracket_type_five_experiment(n_runs=3, m=20)