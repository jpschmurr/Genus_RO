import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List

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
    ref_list = list(set(ref_list))  # Remove duplicates.

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