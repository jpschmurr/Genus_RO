from experiment_functions import billiard_graph_min_genus_experiment, billiard_graph_min_genus_experiment_n_runs, billiard_bracket_type_one_experiment, billiard_bracket_type_two_experiment, billiard_bracket_type_three_experiment, billiard_bracket_type_four_experiment, billiard_bracket_type_five_experiment

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

    #billiard_graph_min_genus_experiment(bracket)
    #billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=3)
    #billiard_bracket_type_one_experiment(n_runs=3, m=20)
    #billiard_bracket_type_two_experiment(n_runs=3, m=20)
    #billiard_bracket_type_three_experiment(n_runs=3, m=20)
    #billiard_bracket_type_four_experiment(n_runs=3, m=20)
    billiard_bracket_type_five_experiment(n_runs=3, m=20)