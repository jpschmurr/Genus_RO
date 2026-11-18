"""
Hill Climbing Algorithm for Finding the Minimum Genus of Cayley Graphs
======================================================================

This module provides a deterministic and recursive hill climbing optimization
framework for finding the minimum genus 2-cell surface embeddings of graphs,
specifically Cayley graphs (such as billiard graphs generated over generalized
dihedral groups and reflection groups).

Mathematical Background
-----------------------
For a connected graph G = (V, E), an orientable 2-cell embedding onto a surface
of genus g is completely determined by a *rotation system* Pi = {pi_v}_{v in V},
where each pi_v is a cyclic permutation of the edges incident to v.

By the Euler-Poincaré formula:
    chi = V - E + F = 2 - 2g  ==>  g = (2 - (V - E + F)) / 2

where:
    - V is the number of vertices,
    - E is the number of edges,
    - F is the number of 2-cell faces (determined as cyclic orbits of sigma o alpha,
      with alpha(u, v) = (v, u) and sigma(v, u) = (v, pi_v(u))).

Neighborhood Operators & Tie Recursion
--------------------------------------
In standard Simulated Annealing, a single random move is sampled per iteration.
In contrast, this Hill Climbing algorithm:
  1. Explores ALL neighbors of the current rotation system (by default, all possible
     single-vertex 2-swaps, reversing segment permutations, or edge-targeted darts).
  2. Identifies all neighbors that achieve a strictly BETTER (lower) genus than the
     current state.
  3. In the case of TIES (where multiple neighbors share the best improved genus):
     Standard hill climbing arbitrarily picks one neighbor, which can get trapped in
     suboptimal local minima. Here, the algorithm RECURSES on ALL tied neighbors,
     exploring all promising branches until each branch reaches its local minimum,
     and returns the overall true minimum genus found across all explored branches.
  4. Utilizes canonical representation and memoization to prune identical rotation
     systems across symmetric or commuting swap sequences, avoiding redundant work.
"""

from __future__ import annotations

import os
import sys
import time
import csv
import math
import random
from typing import Dict, List, Tuple, Set, Optional, Callable, Any

# Ensure local imports work regardless of whether executed from repo root or subfolder
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from face_and_genus_counting_functions import (
        genus_from_rotation,
        random_rotation_from_adj,
        print_face_length_counts,
        face_length_counts,
        count_faces,
        make_sigma,
        make_alpha,
        build_darts,
    )
    from rotation_and_plotting_functions import (
        neighbor_easy_swap,
        neighbor_segment_reverse,
        neighbor_edge_dart,
        neighbor_full_shuffle,
        propose_multi_neighbor,
    )
    from graph_generation_functions import (
        make_billiard_graph_integers,
        graph_info,
    )
except ImportError:
    from Billiards_Experiment.face_and_genus_counting_functions import (
        genus_from_rotation,
        random_rotation_from_adj,
        print_face_length_counts,
        face_length_counts,
        count_faces,
        make_sigma,
        make_alpha,
        build_darts,
    )
    from Billiards_Experiment.rotation_and_plotting_functions import (
        neighbor_easy_swap,
        neighbor_segment_reverse,
        neighbor_edge_dart,
        neighbor_full_shuffle,
        propose_multi_neighbor,
    )
    from Billiards_Experiment.graph_generation_functions import (
        make_billiard_graph_integers,
        graph_info,
    )


# ============================================================================
# CANONICAL REPRESENTATION & ROTATION SYSTEM HELPERS
# ============================================================================

def canonical_cycle(cycle: List[int]) -> Tuple[int, ...]:
    """Rotate cycle list so the minimum element is first (cyclic canonical form)."""
    if not cycle:
        return ()
    min_idx = cycle.index(min(cycle))
    return tuple(cycle[min_idx:] + cycle[:min_idx])


def canonical_rotation(rotation: Dict[int, List[int]]) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """Return an immutable, hashable canonical representation of the rotation system."""
    return tuple((v, canonical_cycle(rotation[v])) for v in sorted(rotation.keys()))


def copy_rotation(rotation: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """Return a deep copy of the rotation system dictionary."""
    return {v: neighs[:] for v, neighs in rotation.items()}


# ============================================================================
# EXHAUSTIVE NEIGHBORHOOD GENERATORS
# ============================================================================

def apply_swap(rotation: Dict[int, List[int]], vertex: int, i: int, j: int) -> Dict[int, List[int]]:
    """Swap neighbors at indices i and j of the given vertex."""
    new_rot = copy_rotation(rotation)
    new_rot[vertex][i], new_rot[vertex][j] = new_rot[vertex][j], new_rot[vertex][i]
    return new_rot


def get_all_swap_neighbors(
    rotation: Dict[int, List[int]],
    deduplicate: bool = True
) -> List[Dict[int, List[int]]]:
    """Generate all distinct 2-swap neighbors across all vertices."""
    seen: Set[Tuple[Tuple[int, Tuple[int, ...]], ...]] = set()
    neighbors: List[Dict[int, List[int]]] = []

    for v in sorted(rotation.keys()):
        m = len(rotation[v])
        if m < 3:
            # For degree < 3, swapping the only 2 elements yields the identical cyclic order
            continue
        for i in range(m):
            for j in range(i + 1, m):
                cand = apply_swap(rotation, v, i, j)
                if deduplicate:
                    c_key = canonical_rotation(cand)
                    if c_key in seen:
                        continue
                    seen.add(c_key)
                neighbors.append(cand)

    return neighbors


def apply_segment_reverse(
    rotation: Dict[int, List[int]],
    vertex: int,
    i: int,
    j: int
) -> Dict[int, List[int]]:
    """Reverse slice [i:j+1] in rotation[vertex]."""
    new_rot = copy_rotation(rotation)
    new_rot[vertex][i:j + 1] = reversed(new_rot[vertex][i:j + 1])
    return new_rot


def get_all_segment_reverse_neighbors(
    rotation: Dict[int, List[int]],
    deduplicate: bool = True
) -> List[Dict[int, List[int]]]:
    """Generate all distinct single-vertex segment reversal neighbors."""
    seen: Set[Tuple[Tuple[int, Tuple[int, ...]], ...]] = set()
    neighbors: List[Dict[int, List[int]]] = []

    for v in sorted(rotation.keys()):
        m = len(rotation[v])
        if m < 3:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                cand = apply_segment_reverse(rotation, v, i, j)
                if deduplicate:
                    c_key = canonical_rotation(cand)
                    if c_key in seen:
                        continue
                    seen.add(c_key)
                neighbors.append(cand)

    return neighbors


def apply_edge_dart(
    rotation: Dict[int, List[int]],
    vertex: int,
    idx: int,
    shift: int
) -> Dict[int, List[int]]:
    """Shift neighbor at idx in rotation[vertex] by shift (+1 or -1)."""
    new_rot = copy_rotation(rotation)
    m = len(new_rot[vertex])
    elem = new_rot[vertex].pop(idx)
    new_rot[vertex].insert((idx + shift) % m, elem)
    return new_rot


def get_all_edge_dart_neighbors(
    rotation: Dict[int, List[int]],
    deduplicate: bool = True
) -> List[Dict[int, List[int]]]:
    """Generate all distinct edge dart neighbors."""
    seen: Set[Tuple[Tuple[int, Tuple[int, ...]], ...]] = set()
    neighbors: List[Dict[int, List[int]]] = []

    for v in sorted(rotation.keys()):
        m = len(rotation[v])
        if m < 3:
            continue
        for idx in range(m):
            for shift in (1, -1):
                cand = apply_edge_dart(rotation, v, idx, shift)
                if deduplicate:
                    c_key = canonical_rotation(cand)
                    if c_key in seen:
                        continue
                    seen.add(c_key)
                neighbors.append(cand)

    return neighbors


def get_all_neighbors(
    rotation: Dict[int, List[int]],
    method: str = "swap",
    deduplicate: bool = True,
    adj: Optional[Dict[int, List[int]]] = None
) -> List[Dict[int, List[int]]]:
    """Dispatch to the specified neighborhood generator."""
    method_lower = method.lower().strip()
    if method_lower in ("swap", "easy_swap", "neighbor_easy_swap"):
        return get_all_swap_neighbors(rotation, deduplicate=deduplicate)
    elif method_lower in ("segment_reverse", "neighbor_segment_reverse"):
        return get_all_segment_reverse_neighbors(rotation, deduplicate=deduplicate)
    elif method_lower in ("edge_dart", "neighbor_edge_dart"):
        return get_all_edge_dart_neighbors(rotation, deduplicate=deduplicate)
    elif method_lower == "all":
        swaps = get_all_swap_neighbors(rotation, deduplicate=False)
        reverses = get_all_segment_reverse_neighbors(rotation, deduplicate=False)
        darts = get_all_edge_dart_neighbors(rotation, deduplicate=False)
        combined = swaps + reverses + darts
        if deduplicate:
            seen: Set[Tuple[Tuple[int, Tuple[int, ...]], ...]] = set()
            unique_list = []
            for c in combined:
                ck = canonical_rotation(c)
                if ck not in seen:
                    seen.add(ck)
                    unique_list.append(c)
            return unique_list
        return combined
    else:
        raise ValueError(
            f"Unknown neighborhood method '{method}'. "
            f"Supported methods: 'swap', 'segment_reverse', 'edge_dart', 'all'."
        )


# ============================================================================
# CORE RECURSIVE HILL CLIMBING ALGORITHM
# ============================================================================

def recursive_hill_climbing(
    adj: Dict[int, List[int]],
    current_rotation: Dict[int, List[int]],
    neighborhood_func: Optional[Callable[[Dict[int, List[int]]], List[Dict[int, List[int]]]]] = None,
    tie_strategy: str = "steepest",
    target_genus: Optional[float] = 0.0,
    max_depth: Optional[int] = None,
    memo: Optional[Dict[Any, Tuple[Dict[int, List[int]], float]]] = None,
    depth: int = 0,
    stats: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Perform recursive hill climbing (steepest descent) to find the minimum genus.

    Algorithm Workflow:
    -------------------
    1. Check memoization cache to avoid re-evaluating identical rotation states.
    2. Compute the genus of current_rotation. If current_genus <= target_genus (e.g. 0.0,
       the absolute theoretical minimum for an orientable embedding), return immediately.
    3. Generate ALL neighbors using neighborhood_func (defaults to get_all_swap_neighbors).
    4. Compute the genus of each neighbor and filter to those with strictly lower genus:
           improving = [N for N in neighbors if genus(N) < current_genus]
    5. If no improving neighbors exist:
       current_rotation is a local minimum. Return (current_rotation, current_genus).
    6. If improving neighbors exist:
       Identify the minimum genus among all neighbors:
           min_g = min(genus(N) for N in improving)
       Identify candidates according to tie_strategy:
           - 'steepest' (default): candidates = [N for N in improving if genus(N) == min_g]
           - 'all_improving': candidates = improving (sorted ascending by genus)
       - If there is a single candidate (no tie):
         Recurse directly on that candidate.
       - In the case of TIES (len(candidates) > 1):
         Recurse on EVERY tied neighbor that has the better genus:
             For each tied candidate N_i:
                 branch_rot, branch_g = recursive_hill_climbing(adj, N_i, ...)
         Select the overall best (minimum genus) found across all branches.
    7. Cache and return the best rotation and genus.

    Parameters
    ----------
    adj : Dict[int, List[int]]
        Adjacency dictionary of the Cayley / billiard graph.
    current_rotation : Dict[int, List[int]]
        The starting rotation system for this search step.
    neighborhood_func : Optional[Callable]
        Function taking a rotation system and returning a list of neighbor rotations.
        Defaults to get_all_swap_neighbors.
    tie_strategy : str, default 'steepest'
        'steepest' : Recurse on all neighbors tied for the minimum genus among improving moves.
        'all_improving' : Recurse on all improving neighbors.
    target_genus : Optional[float], default 0.0
        Early stopping threshold: if a genus <= target_genus is reached, the search halts
        since genus cannot be negative.
    max_depth : Optional[int], default None
        Maximum recursion depth (safety cap).
    memo : Optional[Dict]
        Cache dictionary mapping canonical_rotation -> (best_rotation, best_genus).
    depth : int, default 0
        Current recursion depth in the search tree.
    stats : Optional[Dict]
        Tracking dictionary for profiling metrics (evaluations, ties, cache hits, etc.).
    verbose : bool, default False
        If True, log tie events, depths, and improvements to console.

    Returns
    -------
    Tuple[Dict[int, List[int]], float]
        (best_rotation_system, best_genus_found)
    """
    if memo is None:
        memo = {}
    if stats is None:
        stats = {
            "evaluations": 0,
            "cache_hits": 0,
            "tie_events": 0,
            "branches_explored": 0,
            "max_depth_reached": 0,
        }
    if neighborhood_func is None:
        neighborhood_func = get_all_swap_neighbors

    stats["evaluations"] += 1
    if depth > stats["max_depth_reached"]:
        stats["max_depth_reached"] = depth

    # Check memoization cache to avoid redundant subtree evaluation
    c_key = canonical_rotation(current_rotation)
    if c_key in memo:
        stats["cache_hits"] += 1
        return memo[c_key]

    current_genus = genus_from_rotation(current_rotation, adj)

    # Terminate early if target lower bound reached
    if target_genus is not None and current_genus <= target_genus:
        if verbose:
            print(f"[Depth {depth}] Target genus {target_genus} reached: {current_genus}")
        memo[c_key] = (current_rotation, current_genus)
        return current_rotation, current_genus

    # Depth limit guard
    if max_depth is not None and depth >= max_depth:
        memo[c_key] = (current_rotation, current_genus)
        return current_rotation, current_genus

    # Generate all neighbors
    neighbors = neighborhood_func(current_rotation)

    # Filter to strictly improving neighbors (genus < current_genus)
    improving: List[Tuple[float, Dict[int, List[int]]]] = []
    for cand in neighbors:
        cand_g = genus_from_rotation(cand, adj)
        if cand_g < current_genus:
            improving.append((cand_g, cand))

    # Base case: Local minimum (no improving neighbors)
    if not improving:
        if verbose and depth == 0:
            print(f"[Depth {depth}] Local minimum at genus {current_genus}. No improving neighbors.")
        memo[c_key] = (current_rotation, current_genus)
        return current_rotation, current_genus

    # Determine candidates to recurse on
    min_improving_genus = min(g for g, _ in improving)

    if tie_strategy == "all_improving":
        # Sort candidates ascending by genus so best ones are explored first
        improving.sort(key=lambda item: item[0])
        candidates = [rot for _, rot in improving]
    else:  # 'steepest'
        candidates = [rot for g, rot in improving if g == min_improving_genus]

    num_candidates = len(candidates)
    best_overall_rot = current_rotation
    best_overall_genus = current_genus

    if num_candidates == 1:
        # Unique best neighbor - standard hill climbing step
        cand_rot, cand_g = recursive_hill_climbing(
            adj=adj,
            current_rotation=candidates[0],
            neighborhood_func=neighborhood_func,
            tie_strategy=tie_strategy,
            target_genus=target_genus,
            max_depth=max_depth,
            memo=memo,
            depth=depth + 1,
            stats=stats,
            verbose=verbose,
        )
        if cand_g < best_overall_genus:
            best_overall_genus = cand_g
            best_overall_rot = cand_rot
    else:
        # Tie detected: recurse on all tied improving neighbors
        stats["tie_events"] += 1
        stats["branches_explored"] += num_candidates

        if verbose:
            print(
                f"[Depth {depth}] TIE DETECTED: {num_candidates} neighbors tied with improved genus "
                f"{min_improving_genus} (current: {current_genus}). Recursing on all tied branches..."
            )

        for branch_idx, cand in enumerate(candidates, start=1):
            cand_rot, cand_g = recursive_hill_climbing(
                adj=adj,
                current_rotation=cand,
                neighborhood_func=neighborhood_func,
                tie_strategy=tie_strategy,
                target_genus=target_genus,
                max_depth=max_depth,
                memo=memo,
                depth=depth + 1,
                stats=stats,
                verbose=verbose,
            )

            if cand_g < best_overall_genus:
                best_overall_genus = cand_g
                best_overall_rot = cand_rot

                if verbose:
                    print(
                        f"  [Branch {branch_idx}/{num_candidates} @ depth {depth}] "
                        f"New best found: genus {best_overall_genus}"
                    )

                # Early pruning if absolute theoretical lower bound reached
                if target_genus is not None and best_overall_genus <= target_genus:
                    if verbose:
                        print(f"  [Branch {branch_idx}] Reached target genus {target_genus}; pruning sibling branches.")
                    break

    memo[c_key] = (best_overall_rot, best_overall_genus)
    return best_overall_rot, best_overall_genus


# ============================================================================
# HIGH-LEVEL HILL CLIMBING INTERFACES & RANDOM RESTARTS
# ============================================================================

def hill_climbing_min_genus(
    adj: Dict[int, List[int]],
    init_rotation: Optional[Dict[int, List[int]]] = None,
    neighborhood_func: Optional[Callable] = None,
    tie_strategy: str = "steepest",
    target_genus: Optional[float] = 0.0,
    max_depth: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, List[int]], float, Dict[str, Any]]:
    """
    Run hill climbing to find the minimum genus starting from an initial rotation.

    Parameters
    ----------
    adj : Dict[int, List[int]]
        Adjacency dictionary for the graph.
    init_rotation : Optional[Dict[int, List[int]]]
        Initial rotation system. If None, generated using random_rotation_from_adj(adj, seed=seed).
    neighborhood_func : Optional[Callable]
        Function to generate neighbor rotations. Defaults to get_all_swap_neighbors.
    tie_strategy : str, default 'steepest'
        Strategy for resolving ties: 'steepest' or 'all_improving'.
    target_genus : Optional[float], default 0.0
        Target genus threshold for early termination.
    max_depth : Optional[int], default None
        Recursion depth limit.
    seed : Optional[int], default None
        Random seed for generating the initial rotation system.
    verbose : bool, default False
        Whether to print execution details.

    Returns
    -------
    Tuple[Dict[int, List[int]], float, Dict[str, Any]]
        (best_rotation, best_genus, execution_statistics)
    """
    if init_rotation is None:
        init_rotation = random_rotation_from_adj(adj, seed=seed)

    memo: Dict[Any, Tuple[Dict[int, List[int]], float]] = {}
    stats: Dict[str, Any] = {
        "evaluations": 0,
        "cache_hits": 0,
        "tie_events": 0,
        "branches_explored": 0,
        "max_depth_reached": 0,
    }

    start_time = time.time()
    best_rot, best_genus = recursive_hill_climbing(
        adj=adj,
        current_rotation=init_rotation,
        neighborhood_func=neighborhood_func,
        tie_strategy=tie_strategy,
        target_genus=target_genus,
        max_depth=max_depth,
        memo=memo,
        depth=0,
        stats=stats,
        verbose=verbose,
    )
    stats["elapsed_time_s"] = time.time() - start_time
    stats["initial_genus"] = genus_from_rotation(init_rotation, adj)
    stats["best_genus"] = best_genus

    return best_rot, best_genus, stats


def random_restart_hill_climbing(
    adj: Dict[int, List[int]],
    n_restarts: int = 10,
    seed: Optional[int] = None,
    neighborhood_func: Optional[Callable] = None,
    tie_strategy: str = "steepest",
    target_genus: Optional[float] = 0.0,
    max_depth: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, List[int]], float, List[Dict[str, Any]]]:
    """Run random-restart hill climbing across multiple initial rotations."""
    best_overall_genus = float("inf")
    best_overall_rot: Optional[Dict[int, List[int]]] = None
    records: List[Dict[str, Any]] = []

    for run_idx in range(1, n_restarts + 1):
        run_seed = None if seed is None else seed + run_idx
        init_rot = random_rotation_from_adj(adj, seed=run_seed)
        init_g = genus_from_rotation(init_rot, adj)

        start_time = time.time()
        rot, g, stats = hill_climbing_min_genus(
            adj=adj,
            init_rotation=init_rot,
            neighborhood_func=neighborhood_func,
            tie_strategy=tie_strategy,
            target_genus=target_genus,
            max_depth=max_depth,
            verbose=verbose,
        )
        elapsed = time.time() - start_time

        if g < best_overall_genus:
            best_overall_genus = g
            best_overall_rot = rot

        records.append({
            "run": run_idx,
            "initial_genus": init_g,
            "genus": g,
            "elapsed_time_s": round(elapsed, 3),
            "evaluations": stats["evaluations"],
            "tie_events": stats["tie_events"],
            "max_depth": stats["max_depth_reached"],
            "rotation": rot,
        })

        if target_genus is not None and best_overall_genus <= target_genus:
            if verbose:
                print(f"Target genus {target_genus} achieved on run {run_idx}! Halting restarts.")
            break

    return best_overall_rot, best_overall_genus, records


# ============================================================================
# EXPERIMENT WRAPPERS FOR BILLIARD CAYLEY GRAPHS
# ============================================================================

def save_hill_climbing_results_to_csv(
    records: List[Dict[str, Any]],
    csv_filename: str = "experiment_results.csv"
) -> None:
    """Append experiment records to CSV matching the repository schema."""
    if not records or not csv_filename:
        return

    fieldnames = [
        "bracket",
        "neighbor_function",
        "run",
        "genus",
        "elapsed_time_s",
        "best_genus",
        "total_faces",
        "face_distribution",
        "best_rotation",
    ]

    dir_name = os.path.dirname(csv_filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    file_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def billiard_graph_hill_climbing_experiment(
    bracket: List[int],
    n_restarts: int = 5,
    method: str = "swap",
    tie_strategy: str = "steepest",
    target_genus: Optional[float] = 0.0,
    verbose: bool = False,
    csv_filename: Optional[str] = "experiment_results.csv",
) -> Tuple[Dict[int, List[int]], float]:
    """Run minimum genus hill climbing on a billiard graph bracket."""
    print(f"\n=== Billiard Graph Hill Climbing: Bracket {bracket} ===")
    adj = make_billiard_graph_integers(bracket)
    print("\n=== Graph Information ===")
    graph_info(adj)

    # 2. Setup neighborhood generator
    def neighborhood_fn(rot: Dict[int, List[int]]) -> List[Dict[int, List[int]]]:
        return get_all_neighbors(rot, method=method, deduplicate=True, adj=adj)

    # 3. Initial rotation and sample genus
    sample_rot = random_rotation_from_adj(adj)
    sample_g = genus_from_rotation(sample_rot, adj)
    print(f"\nInitial random rotation genus sample: {sample_g}")
    print_face_length_counts(sample_rot, adj)

    # 4. Run random-restart recursive hill climbing
    operator_name = f"Hill Climbing ({method.capitalize()} with Tie Recursion [{tie_strategy}])"
    print(f"\nRunning {n_restarts} restarts using {operator_name}...")

    total_start = time.time()
    best_rot, best_genus, run_records = random_restart_hill_climbing(
        adj=adj,
        n_restarts=n_restarts,
        neighborhood_func=neighborhood_fn,
        tie_strategy=tie_strategy,
        target_genus=target_genus,
        verbose=verbose,
    )
    total_elapsed = time.time() - total_start

    # 5. Display run-by-run summary
    print(f"\n=== Run Summary ===")
    for rec in run_records:
        print(
            f"Run {rec['run']}: initial genus {rec['initial_genus']} -> "
            f"best genus {rec['genus']} (ties encountered: {rec['tie_events']}, "
            f"evals: {rec['evaluations']}, time: {rec['elapsed_time_s']:.3f}s)"
        )

    print(f"\n{'='*70}")
    print(f"Overall Best Genus Found: {best_genus}")
    print(f"Total Time: {total_elapsed:.2f} seconds")
    print(f"{'='*70}")

    if best_rot is not None:
        print_face_length_counts(best_rot, adj)
        print("\nBest Rotation System:")
        for v in sorted(best_rot):
            print(f"vertex {v}: {best_rot[v]}")

        # 6. Save to CSV
        if csv_filename:
            best_face_counts = face_length_counts(best_rot, adj)
            csv_records: List[Dict[str, Any]] = []
            for rec in run_records:
                csv_records.append({
                    "bracket": str(bracket),
                    "neighbor_function": operator_name,
                    "run": rec["run"],
                    "genus": rec["genus"],
                    "elapsed_time_s": rec["elapsed_time_s"],
                    "best_genus": best_genus,
                    "total_faces": sum(best_face_counts.values()),
                    "face_distribution": str(dict(sorted(best_face_counts.items()))),
                    "best_rotation": str(dict(sorted(best_rot.items()))),
                })
            save_hill_climbing_results_to_csv(csv_records, csv_filename)
            print(f"\nResults successfully saved to {os.path.abspath(csv_filename)}")

    return best_rot, best_genus


def billiard_graph_hill_climbing_experiment_n_runs(
    bracket: List[int],
    n_runs: int = 3,
    method: str = "swap",
    tie_strategy: str = "steepest",
    target_genus: Optional[float] = 0.0,
    csv_filename: Optional[str] = "experiment_results.csv",
) -> Tuple[Dict[int, List[int]], float]:
    """Drop-in alternative to billiard_graph_min_genus_experiment_n_runs."""
    return billiard_graph_hill_climbing_experiment(
        bracket=bracket,
        n_restarts=n_runs,
        method=method,
        tie_strategy=tie_strategy,
        target_genus=target_genus,
        verbose=False,
        csv_filename=csv_filename,
    )


# ============================================================================
# TRIANGULAR BILLIARD GRAPHS & THEORETICAL GENUS (GRZEGRZOLKA ET AL. 2026)
# ============================================================================

def theoretical_genus_triangular(bracket: List[int]) -> int:
    """
    Compute the exact theoretical orientable genus of the Cayley graph of a
    triangular billiard table T(p1, p2, p3) based on Theorem 6 of
    'The Genus of Cayley Graphs on Billiard Surfaces' (Grzegrzolka, McCartney, Schmurr).

    Theorem 6 Statement:
    --------------------
    The genus of G = G(p1, p2, p3) is always 0 or 1.
    In particular, the genus is zero if and only if T(p1, p2, p3) is isosceles
    and n = p1 + p2 + p3 is even.

    Parameters
    ----------
    bracket : List[int]
        A 3-tuple [p1, p2, p3] representing internal angle numerators of the triangle.

    Returns
    -------
    int
        0 if the Cayley graph is planar (isosceles with even n), else 1.
    """
    if len(bracket) != 3:
        raise ValueError(f"Triangular billiard graphs require a 3-tuple bracket, got {bracket}.")
    p1, p2, p3 = bracket
    n = p1 + p2 + p3
    is_isosceles = (p1 == p2 or p2 == p3 or p1 == p3)
    # Genus is 0 if and only if isosceles and n is even
    return 0 if (is_isosceles and n % 2 == 0) else 1


def find_stuck_seeds_triangular(
    bracket: List[int],
    n_seeds: int = 20,
    start_seed: int = 0,
    neighborhood_func: Optional[Callable] = None,
    tie_strategy: str = "steepest",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Test hill climbing across seeds for a triangle and identify sub-optimal local minima."""
    true_g = theoretical_genus_triangular(bracket)
    adj = make_billiard_graph_integers(bracket)

    stuck_seeds: List[int] = []
    successful_seeds: List[int] = []
    records: List[Dict[str, Any]] = []

    for s in range(start_seed, start_seed + n_seeds):
        init_rot = random_rotation_from_adj(adj, seed=s)
        init_g = genus_from_rotation(init_rot, adj)

        rot, found_g, stats = hill_climbing_min_genus(
            adj=adj,
            init_rotation=init_rot,
            neighborhood_func=neighborhood_func,
            tie_strategy=tie_strategy,
            target_genus=float(true_g),
            verbose=verbose,
        )

        is_stuck = (found_g > true_g)
        if is_stuck:
            stuck_seeds.append(s)
        else:
            successful_seeds.append(s)

        records.append({
            "seed": s,
            "initial_genus": init_g,
            "found_genus": found_g,
            "true_genus": true_g,
            "stuck": is_stuck,
            "tie_events": stats["tie_events"],
            "evaluations": stats["evaluations"],
            "elapsed_time_s": stats["elapsed_time_s"],
        })

    return {
        "bracket": bracket,
        "true_genus": true_g,
        "stuck_count": len(stuck_seeds),
        "total_seeds": n_seeds,
        "stuck_seeds": stuck_seeds,
        "successful_seeds": successful_seeds,
        "records": records,
    }


# ============================================================================
# STANDALONE CLI ENTRYPOINT
# ============================================================================

def main() -> None:
    """CLI entrypoint supporting both command-line arguments and interactive prompts."""
    import argparse

    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Recursive Hill Climbing for Cayley Graphs Minimum Genus"
        )
        parser.add_argument(
            "bracket",
            nargs="*",
            type=int,
            help="Billiard bracket tuple as space-separated integers (e.g. 3 3 4 or 2 1 1)",
        )
        parser.add_argument(
            "--seed",
            "-s",
            type=int,
            default=None,
            help="Evaluate a single specific random seed (e.g. --seed 0)",
        )
        parser.add_argument(
            "--restarts",
            "-r",
            type=int,
            default=5,
            help="Number of random restarts (default: 5)",
        )
        parser.add_argument(
            "--method",
            "-m",
            type=str,
            default="swap",
            choices=["swap", "segment_reverse", "edge_dart", "all"],
            help="Neighborhood operator (default: 'swap')",
        )
        parser.add_argument(
            "--check-stuck",
            action="store_true",
            help="Scan seeds and check for local minima traps against Theorem 6 (triangles only)",
        )
        parser.add_argument(
            "--seeds",
            type=int,
            default=20,
            help="Number of seeds to scan when using --check-stuck (default: 20)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Display verbose climb and tie-branching details",
        )
        args = parser.parse_args()

        bracket = args.bracket if args.bracket else [1, 2, 3, 2]

        if args.check_stuck:
            if len(bracket) != 3:
                print(f"Error: --check-stuck requires a triangular bracket (3 integers), got {bracket}.")
                sys.exit(1)
            report = find_stuck_seeds_triangular(bracket, n_seeds=args.seeds, verbose=args.verbose)
            print(f"\n=== Stuck Seed Scan for Triangle {bracket} ===")
            print(f"Theoretical Genus (Theorem 6): {report['true_genus']}")
            print(f"Stuck Seeds ({report['stuck_count']}/{report['total_seeds']}): {report['stuck_seeds']}")
            print("\nSeed Breakdown:")
            for r in report["records"]:
                status = "STUCK (trapped in local minimum)" if r["stuck"] else "FOUND MINIMUM"
                print(f"  Seed {r['seed']:2d}: initial={r['initial_genus']} -> found={r['found_genus']} | {status}")
            return

        if args.seed is not None:
            adj = make_billiard_graph_integers(bracket)
            print(f"\n=== Single Seed Run: Bracket {bracket}, Seed {args.seed} ===")
            graph_info(adj)
            init_rot = random_rotation_from_adj(adj, seed=args.seed)
            init_g = genus_from_rotation(init_rot, adj)
            print(f"Initial rotation genus: {init_g}")

            true_g = theoretical_genus_triangular(bracket) if len(bracket) == 3 else None
            if true_g is not None:
                print(f"Theoretical minimum genus (Theorem 6): {true_g}")

            best_rot, best_g, stats = hill_climbing_min_genus(
                adj=adj,
                init_rotation=init_rot,
                neighborhood_func=lambda r: get_all_neighbors(r, method=args.method, deduplicate=True, adj=adj),
                verbose=args.verbose,
            )

            print(f"\nResult:")
            print(f"  Initial Genus:     {init_g}")
            print(f"  Best Genus Found:  {best_g}")
            print(f"  Evaluations:       {stats['evaluations']}")
            print(f"  Tie Events:        {stats['tie_events']}")
            print(f"  Elapsed Time:      {stats['elapsed_time_s']:.3f}s")

            if true_g is not None:
                if best_g > true_g:
                    print(f"\n>> OUTCOME: STUCK AT LOCAL MINIMUM (genus {best_g} > theoretical minimum {true_g}) <<")
                else:
                    print(f"\n>> OUTCOME: SUCCESS (reached theoretical minimum {true_g}) <<")
            print_face_length_counts(best_rot, adj)
            return

        billiard_graph_hill_climbing_experiment(
            bracket=bracket,
            n_restarts=args.restarts,
            method=args.method,
            tie_strategy="steepest",
            target_genus=0.0,
            verbose=args.verbose,
            csv_filename="experiment_results.csv",
        )
    else:
        print("=" * 70)
        print("  Recursive Hill Climbing for Cayley Graphs Minimum Genus")
        print("=" * 70)

        bracket_input = input(
            "Enter billiard bracket (e.g. '1 2 3 2' or '3 3 4' or '2 1 1'): "
        ).strip()
        bracket = [int(x) for x in bracket_input.split()] if bracket_input else [1, 2, 3, 2]

        seed_input = input("Enter specific seed to test (or press Enter for multi-restart experiment): ").strip()
        if seed_input:
            seed = int(seed_input)
            adj = make_billiard_graph_integers(bracket)
            print(f"\n=== Single Seed Run: Bracket {bracket}, Seed {seed} ===")
            graph_info(adj)
            init_rot = random_rotation_from_adj(adj, seed=seed)
            init_g = genus_from_rotation(init_rot, adj)
            print(f"Initial rotation genus: {init_g}")

            true_g = theoretical_genus_triangular(bracket) if len(bracket) == 3 else None
            if true_g is not None:
                print(f"Theoretical minimum genus (Theorem 6): {true_g}")

            best_rot, best_g, stats = hill_climbing_min_genus(
                adj=adj,
                init_rotation=init_rot,
                verbose=False,
            )
            print(f"\nResult:")
            print(f"  Initial Genus:     {init_g}")
            print(f"  Best Genus Found:  {best_g}")
            print(f"  Evaluations:       {stats['evaluations']}")
            print(f"  Tie Events:        {stats['tie_events']}")
            print(f"  Elapsed Time:      {stats['elapsed_time_s']:.3f}s")
            if true_g is not None:
                if best_g > true_g:
                    print(f"\n>> OUTCOME: STUCK AT LOCAL MINIMUM (genus {best_g} > theoretical minimum {true_g}) <<")
                else:
                    print(f"\n>> OUTCOME: SUCCESS (reached theoretical minimum {true_g}) <<")
            print_face_length_counts(best_rot, adj)
            return

        runs_input = input("Enter number of restarts [default 5]: ").strip()
        n_restarts = int(runs_input) if runs_input else 5

        method_input = input("Operator ('swap', 'segment_reverse', 'edge_dart', 'all') [default 'swap']: ").strip().lower()
        method = method_input if method_input else "swap"

        billiard_graph_hill_climbing_experiment(
            bracket=bracket,
            n_restarts=n_restarts,
            method=method,
            tie_strategy="steepest",
            target_genus=0.0,
            verbose=False,
            csv_filename="experiment_results.csv",
        )


if __name__ == "__main__":
    main()
