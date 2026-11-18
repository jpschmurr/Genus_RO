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

### Case Study 1: $T(3, 3, 4)$ (`bracket = [3, 3, 4]`)
*Featured explicitly in Figures 3 & 4 (pages 4–5) of Grzegrzolka, McCartney, Schmurr.*
- **Properties**: Angles $\frac{3\pi}{10}, \frac{3\pi}{10}, \frac{4\pi}{10}$; $n = 10$; $V = 20, E = 30$; 3-regular; isosceles with $n$ even.
- **Theoretical Genus**: **$0$** (proven in Theorem 6).
- **Stuck Seeds**:
  - **Seed 0**: Initial genus is $3.0$. All 20 swap neighbors have genus $\ge 3.0$ (0 improving moves). The algorithm immediately halts at local minimum **$3.0$** (fails to reach 0).
  - **Seed 1**: Initial genus $5.0 \to$ improves to **$3.0$** and halts (fails to reach 0).
  - **Seed 4**: Initial genus $4.0 \to$ 0 improving moves, halts at **$4.0$** (fails to reach 0).
- **Empirical Severity**: Across **200 random restarts**, hill climbing **never found genus 0** ($0\%$ success rate: 79 trapped at genus 3, 71 at genus 2, 28 at genus 4, 20 at genus 1). In contrast, Simulated Annealing finds genus 0 because it can accept temporary uphill moves.

### Case Study 2: $T(2, 1, 1)$ (`bracket = [2, 1, 1]`)
*The $\pi/2, \pi/4, \pi/4$ right isosceles triangle featured in Figure 2 of the paper.*
- **Properties**: Angles $\frac{2\pi}{4}, \frac{\pi}{4}, \frac{\pi}{4}$; $n = 4$; $V = 8, E = 12$; 3-regular; isosceles with $n$ even.
- **Theoretical Genus**: **$0$** (proven in Theorem 6).
- **Stuck Seeds**:
  - **Seed 0**: Initial genus $2.0 \to$ immediately halts at local minimum **$2.0$** (fails to reach 0).
  - **Seed 2**: Initial genus $2.0 \to$ improves to **$1.0$** and gets trapped (fails to reach 0).
  - **Seed 3**: Initial genus $2.0 \to$ immediately halts at local minimum **$2.0$** (fails to reach 0).

### Case Study 3: $T(1, 2, 4)$ (`bracket = [1, 2, 4]`)
*The $\pi/7, 2\pi/7, 4\pi/7$ scalene triangle.*
- **Properties**: Angles $\frac{\pi}{7}, \frac{2\pi}{7}, \frac{4\pi}{7}$; $n = 7$; $V = 14, E = 21$; 3-regular; scalene (not isosceles).
- **Theoretical Genus**: **$1$** (proven in Theorem 6).
- **Stuck Seeds**:
  - **Seed 0**: Initial genus $3.0 \to$ improves to **$2.0$** and gets trapped (fails to reach 1).
  - **Seed 1**: Initial genus $3.0 \to$ 0 improving moves, immediately halts at **$3.0$** (fails to reach 1).
  - **Seed 3**: Initial genus $3.0 \to$ improves to **$2.0$** and gets trapped (fails to reach 1).

---

## 4. CLI Commands & Entry Instructions for Case Studies

You can reproduce any of the case studies above from your terminal. Both the root wrapper (`python hill_climbing.py`) and the experiment module (`python Billiards_Experiment/hill_climbing.py`) support direct flag arguments and interactive prompt entries.

### 4.1 Direct CLI Flag Commands (Recommended)

Run these commands directly in your PowerShell or Bash terminal:

#### Case Study 1: $T(3, 3, 4)$ (Theoretical Genus = 0)
```powershell
# Test specific stuck seeds (Seed 0, Seed 1, Seed 4)
python hill_climbing.py 3 3 4 --seed 0
python hill_climbing.py 3 3 4 --seed 1
python hill_climbing.py 3 3 4 --seed 4

# Run a batch scan across the first 10 seeds (all 10 get stuck)
python hill_climbing.py 3 3 4 --check-stuck --seeds 10
```

#### Case Study 2: $T(2, 1, 1)$ (Theoretical Genus = 0)
```powershell
# Test specific stuck seeds (Seed 0, Seed 2, Seed 3)
python hill_climbing.py 2 1 1 --seed 0
python hill_climbing.py 2 1 1 --seed 2
python hill_climbing.py 2 1 1 --seed 3

# Run a batch scan across the first 10 seeds
python hill_climbing.py 2 1 1 --check-stuck --seeds 10
```

#### Case Study 3: $T(1, 2, 4)$ (Theoretical Genus = 1)
```powershell
# Test specific stuck seeds (Seed 0, Seed 1, Seed 3)
python hill_climbing.py 1 2 4 --seed 0
python hill_climbing.py 1 2 4 --seed 1
python hill_climbing.py 1 2 4 --seed 3

# Run a batch scan across the first 10 seeds
python hill_climbing.py 1 2 4 --check-stuck --seeds 10
```

---

### 4.2 Interactive Entry Instructions (Prompt-by-Prompt)

If you run `python hill_climbing.py` without arguments, the script enters interactive mode. Follow the prompt instructions below to reproduce the stuck cases:

#### How to test Seed 0 on Bracket `3 3 4`:
1. Start the script:
   ```powershell
   python hill_climbing.py
   ```
2. **Prompt 1**:
   ```text
   Enter billiard bracket (e.g. '1 2 3 2' or '3 3 4' or '2 1 1'):
   ```
   **Entry**: Type `3 3 4` and press `Enter`.
3. **Prompt 2**:
   ```text
   Enter specific seed to test (or press Enter for multi-restart experiment):
   ```
   **Entry**: Type `0` and press `Enter`.
4. **Output displayed**:
   ```text
   === Single Seed Run: Bracket [3, 3, 4], Seed 0 ===
   Number of vertices: 20
   Number of edges:    30
   Degrees:            [3]
   Regularity:         3-regular
   Initial rotation genus: 3.0
   Theoretical minimum genus (Theorem 6): 0

   Result:
     Initial Genus:     3.0
     Best Genus Found:  3.0
     Evaluations:       1
     Tie Events:        0
     Elapsed Time:      0.001s

   >> OUTCOME: STUCK AT LOCAL MINIMUM (genus 3.0 > theoretical minimum 0) <<
   ```

#### How to run a multi-restart experiment interactively:
1. Run `python hill_climbing.py`.
2. At the bracket prompt, enter `3 3 4` (or any bracket).
3. At the seed prompt, **press `Enter`** (leave blank) to choose multi-restart mode.
4. At `Enter number of restarts [default 5]:`, type `5` (or press `Enter`).
5. At `Operator ('swap', 'segment_reverse', 'edge_dart', 'all') [default 'swap']:`, press `Enter` to use single-vertex 2-swaps.

---

### 4.3 Piped Console One-Liners

You can also automate the interactive prompts without entering them manually:

#### Windows PowerShell:
```powershell
# Case Study 1: Bracket [3, 3, 4], Seed 0
"3 3 4`n0" | python hill_climbing.py

# Case Study 2: Bracket [2, 1, 1], Seed 0
"2 1 1`n0" | python hill_climbing.py

# Case Study 3: Bracket [1, 2, 4], Seed 0
"1 2 4`n0" | python hill_climbing.py
```

#### Linux / macOS / Git Bash:
```bash
# Case Study 1: Bracket [3, 3, 4], Seed 0
printf "3 3 4\n0\n" | python hill_climbing.py

# Case Study 2: Bracket [2, 1, 1], Seed 0
printf "2 1 1\n0\n" | python hill_climbing.py

# Case Study 3: Bracket [1, 2, 4], Seed 0
printf "1 2 4\n0\n" | python hill_climbing.py
```

---

## 5. API Reference

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

## 6. Python Scripting Examples

### Basic Hill Climbing on a Billiard Graph
```python
import hill_climbing as hc
from Billiards_Experiment.graph_generation_functions import make_billiard_graph_integers

adj = make_billiard_graph_integers([1, 2, 3, 2])
best_rot, best_genus, stats = hc.hill_climbing_min_genus(adj, seed=42)

print(f"Initial genus: {stats['initial_genus']}")
print(f"Best genus:    {best_genus}")
print(f"Tie events:    {stats['tie_events']}")
```

### Random Restarts
```python
import hill_climbing as hc
from Billiards_Experiment.graph_generation_functions import make_billiard_graph_integers

adj = make_billiard_graph_integers([1, 2, 3, 2])
best_rot, best_genus, records = hc.random_restart_hill_climbing(adj, n_restarts=5)
print(f"Best genus across 5 runs: {best_genus}")
```

### Programmatic Stuck Seed Verification
```python
import hill_climbing as hc

report = hc.find_stuck_seeds_triangular([3, 3, 4], n_seeds=10)
print(f"Theoretical Genus: {report['true_genus']}")
print(f"Stuck Seeds ({report['stuck_count']}/{report['total_seeds']}): {report['stuck_seeds']}")
for r in report['records'][:5]:
    print(f"  Seed {r['seed']}: initial={r['initial_genus']} -> found={r['found_genus']} (stuck={r['stuck']})")
```
