"""Root wrapper and re-exports for the recursive hill climbing algorithm."""

from __future__ import annotations

import os
import sys

# Ensure Billiards_Experiment is accessible in sys.path
_root_dir = os.path.dirname(os.path.abspath(__file__))
_billiards_dir = os.path.join(_root_dir, "Billiards_Experiment")
if _billiards_dir not in sys.path:
    sys.path.insert(0, _billiards_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from Billiards_Experiment.hill_climbing import (
    canonical_cycle,
    canonical_rotation,
    copy_rotation,
    apply_swap,
    get_all_swap_neighbors,
    apply_segment_reverse,
    get_all_segment_reverse_neighbors,
    apply_edge_dart,
    get_all_edge_dart_neighbors,
    get_all_neighbors,
    recursive_hill_climbing,
    hill_climbing_min_genus,
    random_restart_hill_climbing,
    save_hill_climbing_results_to_csv,
    billiard_graph_hill_climbing_experiment,
    billiard_graph_hill_climbing_experiment_n_runs,
    theoretical_genus_triangular,
    find_stuck_seeds_triangular,
)

__all__ = [
    "canonical_cycle",
    "canonical_rotation",
    "copy_rotation",
    "apply_swap",
    "get_all_swap_neighbors",
    "apply_segment_reverse",
    "get_all_segment_reverse_neighbors",
    "apply_edge_dart",
    "get_all_edge_dart_neighbors",
    "get_all_neighbors",
    "recursive_hill_climbing",
    "hill_climbing_min_genus",
    "random_restart_hill_climbing",
    "save_hill_climbing_results_to_csv",
    "billiard_graph_hill_climbing_experiment",
    "billiard_graph_hill_climbing_experiment_n_runs",
    "theoretical_genus_triangular",
    "find_stuck_seeds_triangular",
]

if __name__ == "__main__":
    import Billiards_Experiment.hill_climbing as hc
    hc.main()
