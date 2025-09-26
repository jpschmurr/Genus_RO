import random
import math
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set

# === Graph + rotation utilities ===
def complete_graph_adj(n: int) -> Dict[int, List[int]]:
    """Adjacency for complete graph K_n."""
    return {v: [u for u in range(n) if u != v] for v in range(n)}

def complete_bipartite_adj(m: int, n: int) -> Dict[int, List[int]]:
    """Adjacency for complete bipartite graph K_{m,n}.
       Vertices 0..m-1 = left part, m..m+n-1 = right part."""
    adj: Dict[int, List[int]] = {}
    left = range(m)
    right = range(m, m+n)
    # left side connects to all right side
    for v in left:
        adj[v] = list(right)
    # right side connects to all left side
    for v in right:
        adj[v] = list(left)
    return adj

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

def genus_from_rotation(rotation: Dict[int, List[int]], adj: Dict[int, List[int]]) -> float:
    V = len(adj)
    E = sum(len(neighs) for neighs in adj.values()) // 2
    F = count_faces(rotation, adj)
    chi = V - E + F
    g = (2 - chi) / 2
    return g

# === Theoretical genus formulas ===
def true_genus_complete(n: int) -> int:
    """True orientable genus of complete graph K_n."""
    return math.ceil((n - 3) * (n - 4) / 12)

def true_genus_bipartite(m: int, n: int) -> int:
    """True orientable genus of complete bipartite graph K_{m,n}."""
    return math.ceil((m - 2) * (n - 2) / 4)

# === Neighbor proposals ===
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

# Multi-neighborhood: pick neighbors probabilistically
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

# === Plotting neighbor genus distribution ===
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

# === Simulated annealing ===
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
        candidate = neighbor_func(current) if neighbor_func != propose_multi_neighbor else neighbor_func(current, adj)
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

# === Main program ===
if __name__ == "__main__":
    graph_type = input("Enter graph type ('complete' or 'bipartite'): ").strip().lower()
    if graph_type.startswith("c"):  # complete graph
        n = int(input("Enter n (for K_n): "))
        adj = complete_graph_adj(n)
        true_g = true_genus_complete(n)
        label = f"K_{n}"
    else:  # bipartite
        m = int(input("Enter m (left-side vertices): "))
        n = int(input("Enter n (right-side vertices): "))
        adj = complete_bipartite_adj(m, n)
        true_g = true_genus_bipartite(m, n)
        label = f"K_{{{m},{n}}}"

    # random rotation
    init_rotation = random_rotation_from_adj(adj)
    print(f"\nInitial random rotation system for {label}:")
    for v in sorted(init_rotation):
        print(f"vertex {v}: {init_rotation[v]}")
    init_g = genus_from_rotation(init_rotation, adj)
    print(f"Initial genus: {init_g} (True genus by formula: {true_g})\n")

    neighbors = [
        ("Easy: single-vertex 2-swap", neighbor_easy_swap),
        ("Medium: segment reversal", neighbor_segment_reverse),
        ("Medium: edge-targeted dart", lambda r: neighbor_edge_dart(r, adj)),
        ("Hard: full-vertex shuffle", neighbor_full_shuffle),
        ("Multi-neighborhood (weighted)", propose_multi_neighbor)
    ]

    # Plot distributions for each neighbor type
    print("\nPlotting neighbor genus distributions (200 samples each):")
    for name, func in neighbors:
        plot_neighbor_genus_distribution(adj,
                                         init_rotation,
                                         func,
                                         samples=200,
                                         title=f"{name} neighbor genus distribution")

    # Run SA as before
    for name, func in neighbors:
        start_time = time.time()
        best_rot, best_g = simulated_annealing_min_genus(adj, init_rotation, func,
                                                         max_iters=1200000,
                                                         init_temp=3.0,
                                                         final_temp=1e-5)
        elapsed = time.time() - start_time
        print(f"\n=== Neighbor type: {name} ===")
        print(f"Best genus found: {best_g} (True genus: {true_g}, time: {elapsed:.2f} s)")
        for v in sorted(best_rot):
            print(f"vertex {v}: {best_rot[v]}")

    # === Multi-neighborhood repeated 3 times, track best ===
    print("\n=== Multi-neighborhood 3-run best ===")
    best_overall_genus = float('inf')
    best_overall_rot = None
    times = []
    for run in range(1, 4):
        start_time = time.time()
        rot, g = simulated_annealing_min_genus(adj, init_rotation, propose_multi_neighbor,
                                               max_iters=12000,
                                               init_temp=3.0,
                                               final_temp=1e-5)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"Run {run}: genus {g} (True genus: {true_g}, time {elapsed:.2f}s)")
        if g < best_overall_genus:
            best_overall_genus = g
            best_overall_rot = rot

    print(f"\nBest genus across 3 multi-neighborhood runs: {best_overall_genus} (True genus: {true_g})")
    for v in sorted(best_overall_rot):
        print(f"vertex {v}: {best_overall_rot[v]}")
