import random
import matplotlib.pyplot as plt

from typing import Dict, List
from face_and_genus_counting_functions import genus_from_rotation

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