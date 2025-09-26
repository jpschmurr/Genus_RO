import random
import math
import time
from typing import Dict, List, Tuple, Set

# === Graph + rotation utilities ===
def complete_graph_adj(n: int) -> Dict[int, List[int]]:
    return {v: [u for u in range(n) if u != v] for v in range(n)}

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

# ... main ...

if __name__ == "__main__":
    n = int(input("Enter n (for K_n): "))
    adj = complete_graph_adj(n)

    # random rotation
    init_rotation = random_rotation_from_adj(adj)
    print("\nInitial random rotation system:")
    for v in sorted(init_rotation):
        print(f"vertex {v}: {init_rotation[v]}")
    init_g = genus_from_rotation(init_rotation, adj)
    print(f"Initial genus: {init_g}\n")

    neighbors = [
        ("Easy: single-vertex 2-swap", neighbor_easy_swap),
        ("Medium: segment reversal", neighbor_segment_reverse),
        ("Medium: edge-targeted dart", lambda r: neighbor_edge_dart(r, adj)),
        ("Hard: full-vertex shuffle", neighbor_full_shuffle),
        ("Multi-neighborhood (weighted)", propose_multi_neighbor)
    ]

    for name, func in neighbors:
        start_time = time.time()
        best_rot, best_g = simulated_annealing_min_genus(adj, init_rotation, func,
                                                         max_iters=1200000,
                                                         init_temp=3.0,
                                                         final_temp=1e-5)
        elapsed = time.time() - start_time
        print(f"\n=== Neighbor type: {name} ===")
        print(f"Best genus found: {best_g} (time: {elapsed:.2f} s)")
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
        print(f"Run {run}: genus {g} (time {elapsed:.2f}s)")
        if g < best_overall_genus:
            best_overall_genus = g
            best_overall_rot = rot

    print(f"\nBest genus across 3 multi-neighborhood runs: {best_overall_genus}")
    for v in sorted(best_overall_rot):
        print(f"vertex {v}: {best_overall_rot[v]}")
