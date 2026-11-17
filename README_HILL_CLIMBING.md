# Recursive Hill Climbing for Minimum Genus of Cayley Graphs

A deterministic, exhaustive-neighborhood hill climbing algorithm with **recursive tie branching** and **canonical state memoization** for finding the minimum genus 2-cell surface embeddings of Cayley graphs (including billiard graphs of generalized dihedral and reflection groups).

---

## 1. Mathematical Background

### 1.1 Rotation Systems & Surface Embeddings
For a connected graph $G = (V, E)$, an orientable 2-cell embedding onto a surface of genus $g$ is uniquely specified by a **rotation system** $\Pi = \{\pi_v\}_{v \in V}$, where each $\pi_v$ is a cyclic permutation of edges incident to vertex $v$.

By the Euler-Poincaré formula:
$$\chi = V - E + F = 2 - 2g \implies g = \frac{2 - (V - E + F)}{2}$$

where:
- $V$ is the number of vertices,
- $E$ is the number of edges,
- $F$ is the number of 2-cell faces, computed as cyclic orbits of $\sigma \circ \alpha$ where $\alpha(u, v) = (v, u)$ and $\sigma(v, u) = (v, \pi_v(u))$.

### 1.2 Theoretical Genus of Triangular Billiard Graphs
A rational triangular billiard table $T(p_1, p_2, p_3)$ with internal angles $\alpha_i = \frac{p_i \pi}{n}$ (where $n = p_1 + p_2 + p_3$ and $\gcd(p_1, p_2, p_3) = 1$) generates a 3-regular Cayley graph $G(p_1, p_2, p_3) = \operatorname{Cay}(\{a, b, c\}, D_n)$ with $2n$ vertices and $3n$ edges.

According to **Theorem 6** of *The Genus of Cayley Graphs on Billiard Surfaces* (Grzegrzolka, McCartney, Schmurr):
$$\operatorname{genus}(G(p_1, p_2, p_3)) \in \{0, 1\}$$

- **Genus 0 (Planar)**: If and only if $T(p_1, p_2, p_3)$ is **isosceles** (two angles equal) **and** $n = p_1 + p_2 + p_3$ is **even**.
- **Genus 1 (Toroidal)**: In all other cases ($n$ is odd, or the triangle is scalene / non-isosceles).

---

## 2. Algorithm Overview

Standard local search / simulated annealing typically samples a single random neighbor at each step. This hill climbing algorithm is deterministic and exhaustive:

1. **Exhaustive Neighborhood Generation**:
   Generates all valid local perturbations from the current rotation system (by default, all single-vertex 2-swaps across all vertices).
2. **Canonical Deduplication**:
   Cyclic permutations are shifted so their smallest element is at index 0 (`canonical_cycle`). Rotations that represent the same cyclic order are pruned before evaluation.
3. **Evaluation & Descent**:
   Computes the genus of all neighbors and filters for strictly improving moves ($g_{\text{neighbor}} < g_{\text{current}}$).
4. **Recursive Tie Branching**:
   If multiple neighbors tie for the best improved genus, standard hill climbing picks one arbitrarily and often falls into dead ends. Here, the algorithm **branches and recurses on all tied neighbors**, pursuing each branch to its local optimum and returning the global minimum across all branches.
5. **DAG Memoization**:
   Because genus strictly decreases at each recursive step ($g_{\text{next}} < g_{\text{current}}$), the search space is a finite Directed Acyclic Graph (DAG) with maximum depth bounded by $2 g_{\text{init}}$. Canonical states are memoized to avoid re-evaluating intersecting subtrees when disjoint swaps commute.

---

## 3. Local Minima & Counterexamples (Stuck Seeds)

Because hill climbing is strictly downhill, it can become trapped in local minima where every single-vertex move either leaves the genus unchanged or increases it.

### Case Study: $T(3, 3, 4)$ (`bracket = [3, 3, 4]`)
- **Properties**: Angles $\frac{3\pi}{10}, \frac{3\pi}{10}, \frac{4\pi}{10}$; $n = 10$; isosceles with $n$ even.
- **Theoretical Genus**: **$0$** (proven in Theorem 6; illustrated in Figures 3 & 4 of the paper).
- **Stuck Seeds**:
  - **Seed 0**: Initial genus is $3.0$. All 20 swap neighbors have genus $\ge 3.0$ (0 improving moves). The algorithm immediately halts at local minimum **$3.0$**.
  - **Seed 1**: Initial genus $5.0 \to$ improves to **$3.0$** and halts.
  - **Seed 4**: Initial genus $4.0 \to$ 0 improving moves, halts at **$4.0$**.
- **Empirical Restarts**: In an empirical test of **200 random restarts**, hill climbing **never found genus 0** (79 trapped at genus 3, 71 at genus 2, 28 at genus 4, 20 at genus 1). In contrast, Simulated Annealing discovers genus 0 because it can accept temporary uphill moves.

### Additional Examples:
- **$T(2, 1, 1)$** (Theoretical genus: **0**):
  - **Seed 0**: Halts immediately at genus **2.0**.
  - **Seed 2**: Steps down to genus **1.0** and gets trapped.
- **$T(1, 2, 4)$** (Theoretical genus: **1**):
  - **Seed 0**: Steps down to genus **2.0** and gets trapped.
  - **Seed 1**: Halts immediately at genus **3.0**.

---

## 4. API Reference

All functions are available via `hill_climbing` (root) or `Billiards_Experiment.hill_climbing`.

### Core Optimization
- `recursive_hill_climbing(adj, current_rotation, ...)`: Core recursive steepest descent with tie branching.
- `hill_climbing_min_genus(adj, init_rotation=None, seed=None, ...)`: Runs hill climbing from a given or random initial state.
- `random_restart_hill_climbing(adj, n_restarts=10, ...)`: Multi-restart wrapper exploring independent initial rotations.

### Neighborhood Operators
- `get_all_swap_neighbors(rotation, deduplicate=True)`: All single-vertex 2-swaps.
- `get_all_segment_reverse_neighbors(rotation, deduplicate=True)`: All single-vertex segment reversals.
- `get_all_edge_dart_neighbors(rotation, deduplicate=True)`: All single-vertex dart shifts ($\pm 1$).
- `get_all_neighbors(rotation, method="swap", deduplicate=True, adj=None)`: Unified dispatcher (`'swap'`, `'segment_reverse'`, `'edge_dart'`, or `'all'`).

### Triangular Billiard Graph Verification
- `theoretical_genus_triangular(bracket)`: Returns exact theoretical genus ($0$ or $1$) via Theorem 6.
- `find_stuck_seeds_triangular(bracket, n_seeds=20, ...)`: Runs hill climbing across seeds and reports which seeds got trapped above the theoretical genus.

### Experiment Runners & CSV Logging
- `billiard_graph_hill_climbing_experiment(bracket, n_restarts=5, ...)`: Full experiment for a bracket.
- `billiard_graph_hill_climbing_experiment_n_runs(bracket, n_runs=3, ...)`: Drop-in alternative matching `billiard_graph_min_genus_experiment_n_runs`.
- `save_hill_climbing_results_to_csv(records, csv_filename="experiment_results.csv")`: Logs runs in standard repository format.

---

## 5. Usage Examples

### 5.1 Terminal Execution
Run interactively from the repository root:
```powershell
python hill_climbing.py
```
Or from `Billiards_Experiment`:
```powershell
python Billiards_Experiment/hill_climbing.py
```

### 5.2 Python Scripting

#### Basic Hill Climbing on a Billiard Graph
```python
import hill_climbing as hc
from Billiards_Experiment.graph_generation_functions import make_billiard_graph_integers

adj = make_billiard_graph_integers([1, 2, 3, 2])
best_rot, best_genus, stats = hc.hill_climbing_min_genus(adj, seed=42)

print(f"Initial genus: {stats['initial_genus']}")
print(f"Best genus:    {best_genus}")
print(f"Tie events:    {stats['tie_events']}")
```

#### Random Restarts
```python
import hill_climbing as hc
from Billiards_Experiment.graph_generation_functions import make_billiard_graph_integers

adj = make_billiard_graph_integers([1, 2, 3, 2])
best_rot, best_genus, records = hc.random_restart_hill_climbing(adj, n_restarts=5)
print(f"Best genus across 5 runs: {best_genus}")
```

#### Testing Stuck Seeds against Theorem 6
```python
import hill_climbing as hc

report = hc.find_stuck_seeds_triangular([3, 3, 4], n_seeds=10)
print(f"Theoretical Genus: {report['true_genus']}")
print(f"Stuck Seeds ({report['stuck_count']}/{report['total_seeds']}): {report['stuck_seeds']}")
for r in report['records'][:5]:
    print(f"  Seed {r['seed']}: initial={r['initial_genus']} -> found={r['found_genus']} (stuck={r['stuck']})")
```
