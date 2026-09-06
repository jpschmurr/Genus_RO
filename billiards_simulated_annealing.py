import random
import math
import time
import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List, Tuple, Set


# ============================================================
# BILLIARD GRAPH CREATION
# ============================================================

def make_billiard_graph_integers(bracket: List[int]) -> Dict[int, List[int]]:
    """
    Create the adjacency dictionary for a billiard graph.

    Parameters
    ----------
    bracket : list[int]
        An admissible k-tuple of integers.

    Returns
    -------
    Dict[int, List[int]]
        Adjacency dictionary compatible with the genus experiment.

    If k = len(bracket), the graph is k-regular and has 2n vertices,
    where

        n = sum(bracket) / (k - 2).

    The vertices are divided into two parts:

        0, ..., n-1
        n, ..., 2n-1

    and every edge goes between the two parts.
    """

    k = len(bracket)

    if k < 3:
        raise ValueError("The bracket must contain at least 3 integers.")

    if any(b <= 0 for b in bracket):
        raise ValueError("All bracket entries must be positive integers.")

    total = sum(bracket)

    # Check that n is actually an integer.
    if total % (k - 2) != 0:
        raise ValueError(
            f"Invalid bracket {bracket}: "
            f"sum(bracket) = {total} is not divisible by k-2 = {k-2}."
        )

    n = total // (k - 2)

    # Adjacency dictionary.
    adj: Dict[int, List[int]] = {
        v: [] for v in range(2 * n)
    }

    # --------------------------------------------------------
    # Calculate the reference offsets.
    #
    # ref_list[0] = 0
    #
    # ref_list[p] =
    #     (ref_list[p-1] + n - bracket[p]) mod n
    # --------------------------------------------------------

    ref_list = [0]

    for p in range(1, k):
        next_ref = (ref_list[p - 1] + n - bracket[p]) % n
        ref_list.append(next_ref)

    # --------------------------------------------------------
    # Add edges.
    # --------------------------------------------------------

    for rotation_vertex in range(n):

        for a in ref_list:

            flip_vertex = ((a + rotation_vertex) % n) + n

            adj[rotation_vertex].append(flip_vertex)
            adj[flip_vertex].append(rotation_vertex)

    return adj


def billiard_graph_networkx(bracket: List[int]) -> nx.Graph:
    """
    Optional convenience function.

    Creates the billiard graph as a NetworkX graph.
    """

    adj = make_billiard_graph_integers(bracket)

    B = nx.Graph()

    for v in adj:
        B.add_node(v)

    for v, neighbors in adj.items():
        for u in neighbors:
            B.add_edge(v, u)

    return B


# ============================================================
# GRAPH INFORMATION
# ============================================================

def graph_info(adj: Dict[int, List[int]]) -> None:
    """Print basic information about the graph."""

    V = len(adj)
    E = sum(len(neighbors) for neighbors in adj.values()) // 2
    degrees = [len(neighbors) for neighbors in adj.values()]

    print(f"Number of vertices: {V}")
    print(f"Number of edges:    {E}")
    print(f"Degrees:            {sorted(set(degrees))}")

    if len(set(degrees)) == 1:
        print(f"Regularity:         {degrees[0]}-regular")
    else:
        print("Regularity:         Not regular")


# ============================================================
# ROTATION SYSTEM UTILITIES
# ============================================================

def random_rotation_from_adj(adj: Dict[int, List[int]], seed: int = None) -> Dict[int, List[int]]:
    if seed is not None:
        random.seed(seed)
    rot = {v: list(neighs)[:] for v, neighs in adj.items()}
    for v in rot:
        random.shuffle(rot[v])
    return rot


def build_darts(adj: Dict[int, List[int]]) -> List[Tuple[int,int]]:
    darts = []
    for v, neighs in adj.items():
        for u in neighs:
            darts.append((v, u))
    return darts


def make_sigma(rotation: Dict[int, List[int]]) -> Dict[Tuple[int,int], Tuple[int,int]]:
    sigma = {}
    for v, neighs in rotation.items():
        m = len(neighs)
        if m == 0:
            continue
        for i, u in enumerate(neighs):
            nxt = neighs[(i+1) % m]
            sigma[(v,u)] = (v, nxt)
    return sigma


def make_alpha(adj: Dict[int, List[int]]) -> Dict[Tuple[int,int], Tuple[int,int]]:
    alpha = {}
    for v, neighs in adj.items():
        for u in neighs:
            alpha[(v,u)] = (u,v)
    return alpha

# ============================================================
# FACE COUNTING
# ============================================================

def count_faces(rotation: Dict[int, List[int]], adj: Dict[int, List[int]]) -> int:
    sigma = make_sigma(rotation)
    alpha = make_alpha(adj)
    darts = build_darts(adj)
    visited: Set[Tuple[int,int]] = set()
    faces = 0
    for d in darts:
        if d in visited:
            continue
        faces += 1
        cur = d
        while True:
            visited.add(cur)
            cur = sigma[alpha[cur]]
            if cur in visited:
                break
    return faces


# ============================================================
# GENUS
# ============================================================

def genus_from_rotation(rotation: Dict[int, List[int]], adj: Dict[int, List[int]]) -> float:
    V = len(adj)
    E = sum(len(neighs) for neighs in adj.values()) // 2
    F = count_faces(rotation, adj)
    chi = V - E + F
    g = (2 - chi) / 2
    return g


# ============================================================
# NEIGHBOR OPERATORS
# ============================================================

def neighbor_easy_swap(rotation: Dict[int, List[int]]) -> Dict[int, List[int]]:
    # Easy: single-vertex 2-swap
    new_rot = {v: neighs[:] for v, neighs in rotation.items()}
    v = random.choice(list(new_rot.keys()))
    m = len(new_rot[v])
    if m >= 2:
        i, j = random.sample(range(m), 2)
        new_rot[v][i], new_rot[v][j] = new_rot[v][j], new_rot[v][i]
    return new_rot

def neighbor_segment_reverse(rotation: Dict[int, List[int]]) -> Dict[int, List[int]]:
    # Medium: single-vertex segment reversal
    new_rot = {v: neighs[:] for v, neighs in rotation.items()}
    v = random.choice(list(new_rot.keys()))
    m = len(new_rot[v])
    if m >= 2:
        i, j = sorted(random.sample(range(m), 2))
        new_rot[v][i:j+1] = reversed(new_rot[v][i:j+1])
    return new_rot

def neighbor_edge_dart(rotation: Dict[int, List[int]], adj: Dict[int, List[int]]) -> Dict[int, List[int]]:
    # Medium: edge-targeted dart move
    new_rot = {v: neighs[:] for v, neighs in rotation.items()}
    v = random.choice(list(new_rot.keys()))
    if len(new_rot[v]) >= 2:
        u = random.choice(new_rot[v])
        idx = new_rot[v].index(u)
        shift = random.choice([1, -1])
        new_rot[v].insert((idx + shift) % len(new_rot[v]), new_rot[v].pop(idx))
    return new_rot

def neighbor_full_shuffle(rotation: Dict[int, List[int]]) -> Dict[int, List[int]]:
    # Hard: full-vertex shuffle
    new_rot = {v: neighs[:] for v, neighs in rotation.items()}
    v = random.choice(list(new_rot.keys()))
    random.shuffle(new_rot[v])
    return new_rot

def propose_multi_neighbor(rotation: Dict[int, List[int]], adj: Dict[int, List[int]]) -> Dict[int, List[int]]:
    p = random.random()
    if p < 0.5:
        return neighbor_easy_swap(rotation)
    elif p < 0.75:
        return neighbor_segment_reverse(rotation)
    elif p < 0.9:
        return neighbor_edge_dart(rotation, adj)
    else:
        return neighbor_full_shuffle(rotation)


# ============================================================
# PLOT NEIGHBOR GENUS DISTRIBUTION
# ============================================================

def plot_neighbor_genus_distribution(adj,
                                     base_rotation: Dict[int, List[int]],
                                     neighbor_func,
                                     samples: int = 200,
                                     title: str = 'Neighbor genus distribution'):
    """
    Sample `samples` neighbors from `base_rotation` using neighbor_func and plot a histogram of their genus.
    """
    genus_values = []
    for _ in range(samples):
        try:
            cand = neighbor_func(base_rotation)
        except TypeError:
            # if neighbor_func needs adj as well (like edge-dart or multi)
            cand = neighbor_func(base_rotation, adj)
        g = genus_from_rotation(cand, adj)
        genus_values.append(g)

    plt.figure(figsize=(8, 4))
    plt.hist(genus_values, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Genus")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# ============================================================
# SIMULATED ANNEALING
# ============================================================

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
                if best_genus <= 0:
                    break
        temp *= cooling

    return best_rotation, best_genus


# ============================================================
# MAIN PROGRAM
# ============================================================

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

    print(f"\nBracket: {bracket}")

    # --------------------------------------------------------
    # Construct billiard graph.
    # --------------------------------------------------------

    adj = make_billiard_graph_integers(bracket)

    # --------------------------------------------------------
    # Display graph information.
    # --------------------------------------------------------

    print("\n=== Graph Information ===")

    graph_info(adj)

    # Calculate k and n for display.
    k = len(bracket)
    n = sum(bracket) // (k - 2)

    print(f"n:                   {n}")
    print(f"Expected vertices:   {2 * n}")
    print(f"Expected degree:     {k}")

    # --------------------------------------------------------
    # Create random initial rotation system.
    # --------------------------------------------------------

    init_rotation = random_rotation_from_adj(adj)

    print("\n=== Initial Rotation System ===")

    for v in sorted(init_rotation):

        print(
            f"vertex {v}: "
            f"{init_rotation[v]}"
        )

    init_g = genus_from_rotation(
        init_rotation,
        adj
    )

    print(
        f"\nInitial genus: {init_g}"
    )

    # --------------------------------------------------------
    # Neighbor strategies.
    # --------------------------------------------------------

    neighbors = [
        (
            "Easy: single-vertex 2-swap",
            neighbor_easy_swap
        ),
        (
            "Medium: segment reversal",
            neighbor_segment_reverse
        ),
        (
            "Medium: edge-targeted dart",
            neighbor_edge_dart
        ),
        (
            "Hard: full-vertex shuffle",
            neighbor_full_shuffle
        ),
        (
            "Multi-neighborhood (weighted)",
            propose_multi_neighbor
        )
    ]

    # --------------------------------------------------------
    # Plot neighborhood distributions.
    # --------------------------------------------------------

    print(
        "\n=== Neighbor Genus Distributions ==="
    )

    for name, func in neighbors:

        plot_neighbor_genus_distribution(
            adj,
            init_rotation,
            func,
            samples=200,
            title=f"{name} neighbor genus distribution"
        )

    # --------------------------------------------------------
    # Run simulated annealing.
    # --------------------------------------------------------

    for name, func in neighbors:

        start_time = time.time()

        best_rot, best_g = (
            simulated_annealing_min_genus(
                adj,
                init_rotation,
                func,
                max_iters=1200000,
                init_temp=3.0,
                final_temp=1e-5
            )
        )

        elapsed = time.time() - start_time

        print(
            f"\n=== Neighbor type: {name} ==="
        )

        print(
            f"Best genus found: {best_g}"
        )

        print(
            f"Time: {elapsed:.2f} seconds"
        )

        for v in sorted(best_rot):

            print(
                f"vertex {v}: "
                f"{best_rot[v]}"
            )

    # --------------------------------------------------------
    # Multi-neighborhood repeated runs.
    # --------------------------------------------------------

    print(
        "\n=== Multi-neighborhood 3-run best ==="
    )

    best_overall_genus = float("inf")
    best_overall_rot = None

    for run in range(1, 4):

        start_time = time.time()

        rot, g = (
            simulated_annealing_min_genus(
                adj,
                init_rotation,
                propose_multi_neighbor,
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
        f"\nBest genus across 3 runs: "
        f"{best_overall_genus}"
    )

    print("\nBest rotation system:")

    for v in sorted(best_overall_rot):

        print(
            f"vertex {v}: "
            f"{best_overall_rot[v]}"
        )