import random
import math
import time

from typing import Dict, List, Tuple

from graph_generation_functions import make_billiard_graph_integers, graph_info
from rotation_and_plotting_functions import neighbor_easy_swap, neighbor_segment_reverse, neighbor_edge_dart, neighbor_full_shuffle, propose_multi_neighbor
from face_and_genus_counting_functions import genus_from_rotation, random_rotation_from_adj, print_face_length_counts

#----------------------------------------------------------
# Global variable to control display of progress during simulated annealing.
display_progress = False
#----------------------------------------------------------

#===========================================================
# SIMULATED ANNEALING FOR MINIMUM GENUS
#===========================================================

def simulated_annealing_min_genus(adj: Dict[int, List[int]],
                                  init_rotation: Dict[int, List[int]],
                                  neighbor_func,
                                  max_iters: int = 200000,
                                  init_temp: float = 5.0,
                                  final_temp: float = 1e-4) -> Tuple[Dict[int,List[int]], float]:
    current = {v: neighs[:] for v, neighs in init_rotation.items()}
    current_genus = genus_from_rotation(current, adj)
    best_rotation, best_genus = current, current_genus

    temp = init_temp
    cooling = (final_temp / init_temp) ** (1.0 / max_iters) if max_iters > 0 else 1.0

    for it in range(max_iters):
        if neighbor_func in (neighbor_edge_dart, propose_multi_neighbor):
            candidate = neighbor_func(current, adj)
        else:
            candidate = neighbor_func(current)
        cand_genus = genus_from_rotation(candidate, adj)
        delta = cand_genus - current_genus

        if delta <= 0 or random.random() < math.exp(-delta / max(1e-12, temp)):
            current, current_genus = candidate, cand_genus
            if current_genus < best_genus:
                best_genus = current_genus
                best_rotation = current

                #Display useful information about the current best solution
                if display_progress == True:
                    print(f"Iteration {it}: \nNew best genus found: {best_genus}")

                    print_face_length_counts(
                        best_rotation,
                        adj
                    )

                    print(f"\nCurrent Rotation System:")
                    for v in sorted(best_rotation):
                        print(
                            f"vertex {v}: "
                            f"{best_rotation[v]}"
                        )
                    print("\n")

                if best_genus <= 0:
                    break
        temp *= cooling

    return best_rotation, best_genus

#===========================================================
# BILLIARD GRAPH MINIMUM GENUS EXPERIMENT
#===========================================================

def billiard_graph_min_genus_experiment(bracket: List[int],
                                     max_iters: int = 2000000,
                                     init_temp: float = 3.0,
                                     final_temp: float = 1e-5) -> None:
    """
    Run the billiard graph minimum genus experiment for a given bracket.

    Returns the best rotation system and its genus.
    """

    print(f"\n=== Billiard Graph Minimum Genus Experiment for Bracket: {bracket} ===")

    # Construct billiard graph.
    adj = make_billiard_graph_integers(bracket)

    # Display graph information.
    print("\n=== Graph Information ===")
    graph_info(adj)

    # Create random initial rotation system.
    init_rotation = random_rotation_from_adj(adj)

    print("\n=== Initial Rotation System ===")
    for v in sorted(init_rotation):
        print(f"vertex {v}: {init_rotation[v]}")

    init_g = genus_from_rotation(init_rotation, adj)
    print(f"\nInitial genus: {init_g}")
    print_face_length_counts(init_rotation, adj)

    neighbors = [
        ("Easy: single-vertex 2-swap", neighbor_easy_swap),
        #("Medium: segment reversal", neighbor_segment_reverse),
        #("Medium: edge-targeted dart", neighbor_edge_dart),
        #("Hard: full-vertex shuffle", neighbor_full_shuffle),
        #("Multi-neighborhood (weighted)", propose_multi_neighbor)
    ]

    for name, func in neighbors:
    
        start_time = time.time()

        best_rot, best_g = (
            simulated_annealing_min_genus(
                adj,
                init_rotation,
                func,
                max_iters=2000000,
                init_temp=3.0,
                final_temp=1e-5
            )
        )

        elapsed = time.time() - start_time

    
        print(f"\n=== Results for {name} ===")
        print(f"Elapsed time: {elapsed:.2f} seconds")

        print(f"\nBest genus found: {best_g}")
        print_face_length_counts(best_rot, adj)

        print("\nBest rotation system:")
        for v in sorted(best_rot):
            print(f"vertex {v}: {best_rot[v]}")

#============================================================
# BILLIARD GRAPH MINIMUM GENUS EXPERIMENT WITH MULTIPLE RUNS
#============================================================

def billiard_graph_min_genus_experiment_n_runs(bracket: List[int], 
                                                n_runs: int = 3, 
                                                max_iterations: int = 2000000,
                                                init_temperature: float = 3.0,
                                                final_temperature: float = 1e-5, 
                                                funcs: list = [("Easy: Single-vertex 2-swap", neighbor_easy_swap)]) -> None:
    """
    Run the billiard graph minimum genus experiment with a specified neighbor function for a given bracket.
    """
    print(
        f"\n=== {bracket} {n_runs}-run best ==="
    )

    best_overall_genus = float("inf")
    best_overall_rot = None

    adj = make_billiard_graph_integers(bracket)

    init_rotation = random_rotation_from_adj(adj)

    for func in funcs:

        print(f"\n=== {func[0]} ===")
        
        for run in range(1, n_runs + 1):
        
            start_time = time.time()

            rot, g = (
                simulated_annealing_min_genus(
                    adj,
                    init_rotation,
                    func[1],
                    max_iters=max_iterations,
                    init_temp=init_temperature,
                    final_temp=final_temperature
                )
            )

            elapsed = time.time() - start_time

            print(
                f"Run {run}: "
                f"genus {g}, "
                f"time {elapsed:.2f}s"
            )

            if g < best_overall_genus:

                best_overall_genus = g

                best_overall_rot = rot

    print(
        f"\nBest genus across {n_runs} runs: "
        f"{best_overall_genus}"
    )

    print_face_length_counts(
        best_overall_rot,
        adj
    )

    print("\nBest rotation system:")

    for v in sorted(best_overall_rot):

        print(
            f"vertex {v}: "
            f"{best_overall_rot[v]}"
        )

def billiard_graph_min_genus_experiment_n_runs_outdated(bracket: List[int], n_runs: int = 3, func: callable = neighbor_easy_swap) -> None:
    """
    Run the billiard graph minimum genus experiment with a specified neighbor function for a given bracket.
    """
    print(
        f"\n=== {func.__name__} {n_runs}-run best ==="
    )

    best_overall_genus = float("inf")
    best_overall_rot = None

    adj = make_billiard_graph_integers(bracket)

    init_rotation = random_rotation_from_adj(adj)


    for run in range(1, n_runs + 1):

        start_time = time.time()

        rot, g = (
            simulated_annealing_min_genus(
                adj,
                init_rotation,
                func,
                max_iters=12000,
                init_temp=3.0,
                final_temp=1e-5
            )
        )

        elapsed = time.time() - start_time

        print(
            f"Run {run}: "
            f"genus {g}, "
            f"time {elapsed:.2f}s"
        )

        if g < best_overall_genus:

            best_overall_genus = g

            best_overall_rot = rot

    print(
        f"\nBest genus across {n_runs} runs: "
        f"{best_overall_genus}"
    )

    print_face_length_counts(
        best_overall_rot,
        adj
    )

    print("\nBest rotation system:")

    for v in sorted(best_overall_rot):

        print(
            f"vertex {v}: "
            f"{best_overall_rot[v]}"
        )


def billiard_bracket_type_one_experiment(n_runs: int = 3, m: int = 20) -> None:
    """
    Run the billiard graph minimum genus experiment for a bracket of the form [1,1,1,2k-3] for k from 2 to m.
    """
    print("\n=== Finding minimum genus for brackets of the form [1,1,1,2k-3] ===")

    for k in range(2, m + 1):
    
        bracket = [1, 1, 1, 2*k-3]

        print(f"\n=== Bracket: {bracket} ===")

        # Construct billiard graph.
        adj = make_billiard_graph_integers(bracket)

        # Create random initial rotation system.
        init_rotation = random_rotation_from_adj(adj)

        init_g = genus_from_rotation(init_rotation, adj)
        print(f"\nInitial genus: {init_g}")
        print_face_length_counts(init_rotation, adj)

        neighbors = [
            ("Easy: single-vertex 2-swap", neighbor_easy_swap),
            ("Medium: segment reversal", neighbor_segment_reverse),
            ("Medium: edge-targeted dart", neighbor_edge_dart),
            ("Hard: full-vertex shuffle", neighbor_full_shuffle),
            ("Multi-neighborhood (weighted)", propose_multi_neighbor)
        ]

        for func in neighbors:
            billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=n_runs, func=func)


def billiard_bracket_type_two_experiment(n_runs: int = 3, m: int = 20) -> None:
    """
    Run the billiard graph minimum genus experiment for a bracket of the form [1,2,3,2k] for k from 2 to m.
    """
    print("\n=== Finding minimum genus for brackets of the form [1,2,3,2k] ===")

    for k in range(2, m + 1):

        bracket = [1, 2, 3, 2*k]

        print(f"\n=== Bracket: {bracket} ===")

        # Construct billiard graph.
        adj = make_billiard_graph_integers(bracket)

        # Create random initial rotation system.
        init_rotation = random_rotation_from_adj(adj)

        init_g = genus_from_rotation(init_rotation, adj)
        print(f"\nInitial genus: {init_g}")
        print_face_length_counts(init_rotation, adj)

        neighbors = [
            ("Easy: single-vertex 2-swap", neighbor_easy_swap),
            ("Medium: segment reversal", neighbor_segment_reverse),
            ("Medium: edge-targeted dart", neighbor_edge_dart),
            ("Hard: full-vertex shuffle", neighbor_full_shuffle),
            ("Multi-neighborhood (weighted)", propose_multi_neighbor)
        ]

        for func in neighbors:
            billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=n_runs, func=func)

def billiard_bracket_type_three_experiment(n_runs: int = 3, m: int = 20) -> None:
    """
    Run the billiard graph minimum genus experiment for a bracket of the form [1,2,3,4,3k+2] for k from 0 to m.
    """
    print("\n=== Finding minimum genus for brackets of the form [1,2,3,4,3k+2] ===")

    for k in range(0, m + 1):
    
        bracket = [1, 2, 3, 4, 3*k+2]

        print(f"\n=== Bracket: {bracket} ===")

        # Construct billiard graph.
        adj = make_billiard_graph_integers(bracket)

        # Create random initial rotation system.
        init_rotation = random_rotation_from_adj(adj)

        init_g = genus_from_rotation(init_rotation, adj)
        print(f"\nInitial genus: {init_g}")
        print_face_length_counts(init_rotation, adj)

        neighbors = [
            ("Easy: single-vertex 2-swap", neighbor_easy_swap),
            ("Medium: segment reversal", neighbor_segment_reverse),
            ("Medium: edge-targeted dart", neighbor_edge_dart),
            ("Hard: full-vertex shuffle", neighbor_full_shuffle),
            ("Multi-neighborhood (weighted)", propose_multi_neighbor)
        ]

        for func in neighbors:
            billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=n_runs, func=func)

def billiard_bracket_type_four_experiment(n_runs: int = 3, m: int = 20) -> None:

    print("\n=== Finding minimum genus for brackets of the form [k,k+1,k+2,k+3] ===")
        
    for k in range(0, m + 1):
        
        bracket = [k, k+1, k+2, k+3]

        print(f"\n=== Bracket: {bracket} ===")

        # Construct billiard graph.
        adj = make_billiard_graph_integers(bracket)

        # Create random initial rotation system.
        init_rotation = random_rotation_from_adj(adj)

        init_g = genus_from_rotation(init_rotation, adj)
        print(f"\nInitial genus: {init_g}")
        print_face_length_counts(init_rotation, adj)

        neighbors = [
            ("Easy: single-vertex 2-swap", neighbor_easy_swap),
            ("Medium: segment reversal", neighbor_segment_reverse),
            ("Medium: edge-targeted dart", neighbor_edge_dart),
            ("Hard: full-vertex shuffle", neighbor_full_shuffle),
            ("Multi-neighborhood (weighted)", propose_multi_neighbor)
        ]

        for func in neighbors:
            billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=n_runs, func=func[1])

def billiard_bracket_type_five_experiment(n_runs: int = 3, m: int = 20) -> None:
    print("\n=== Finding minimum genus for brackets of the form [1,1,1,1, 3k-4] ===")
    
    for k in range(2, 21):

        bracket = [1, 1, 1, 1, 3*k-4]

        print(f"\n=== Bracket: {bracket} ===")

        adj = make_billiard_graph_integers(bracket)

        init_rotation = random_rotation_from_adj(adj)
        
        # Construct billiard graph.
        adj = make_billiard_graph_integers(bracket)

        # Create random initial rotation system.
        init_rotation = random_rotation_from_adj(adj)

        init_g = genus_from_rotation(init_rotation, adj)
        print(f"\nInitial genus: {init_g}")
        print_face_length_counts(init_rotation, adj)

        neighbors = [
            ("Easy: single-vertex 2-swap", neighbor_easy_swap),
            ("Medium: segment reversal", neighbor_segment_reverse),
            ("Medium: edge-targeted dart", neighbor_edge_dart),
            ("Hard: full-vertex shuffle", neighbor_full_shuffle),
            ("Multi-neighborhood (weighted)", propose_multi_neighbor)
        ]

        for func in neighbors:
            billiard_graph_min_genus_experiment_n_runs(bracket, n_runs=n_runs, func=func[1])

