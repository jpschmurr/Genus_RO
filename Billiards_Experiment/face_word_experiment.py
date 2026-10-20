"""Anneal a billiard graph and report its rotation and face words.

Run from the repository root (paste as one line in PowerShell), for example::

    python Billiards_Experiment/face_word_experiment.py 1 2 3 2 --max-iters 200000 --init-temp 3 --final-temp 0.00001 --seed 42

Letters name distinct reflection offsets in first-occurrence bracket order,
starting with offset 0 as 'a'. An edge from rotation vertex r to reflection
vertex n+s has label (s-r) mod n. Both directions carry the same letter.
Face traversal follows Section 3.2 of "The Genus of Cayley Graphs on Billiard
Surfaces": on arrival, follow the successor of the incoming neighbor.
Words are unreduced boundary words, displayed as their lexicographically
smallest cyclic rotation, with the vertex circuit shifted to match.
Reversing the traversal reverses the word (reflections are involutions).
"""

import argparse
import math
import time
from dataclasses import dataclass
from string import ascii_lowercase
from typing import Callable, Dict, List, Tuple

from experiment_functions import simulated_annealing_min_genus
from face_and_genus_counting_functions import (
    make_alpha,
    make_sigma,
    random_rotation_from_adj,
)
from graph_generation_functions import make_billiard_graph_integers
from rotation_and_plotting_functions import (
    neighbor_easy_swap,
    neighbor_edge_dart,
    neighbor_full_shuffle,
    neighbor_segment_reverse,
    propose_multi_neighbor,
)

Rotation = Dict[int, List[int]]
Dart = Tuple[int, int]

NEIGHBOR_METHODS = {
    "easy_swap": neighbor_easy_swap,
    "segment_reverse": neighbor_segment_reverse,
    "edge_dart": neighbor_edge_dart,
    "full_shuffle": neighbor_full_shuffle,
    "multi": propose_multi_neighbor,
}


@dataclass
class FaceCircuit:
    darts: List[Dart]
    word: str

    @property
    def vertices(self) -> List[int]:
        """Closed vertex walk, including the repeated starting vertex."""
        return [u for u, _ in self.darts] + [self.darts[-1][1]]


@dataclass
class ExperimentResult:
    adjacency: Rotation
    rotation: Rotation
    genus: float
    reflection_labels: Dict[int, str]
    faces: List[FaceCircuit]


def reflection_labels_from_bracket(bracket: List[int]) -> Dict[int, str]:
    """Map distinct offsets to letters without the generator's set reordering.

    As in the existing graph generator, parallel/repeated reflections are
    identified, rather than represented by separately labeled parallel edges.
    Up to 26 distinct reflections are supported so each label is one letter.
    """
    if (len(bracket) < 3
            or any(type(value) is not int or value <= 0 for value in bracket)):
        raise ValueError("The bracket must contain at least three positive integers.")
    if sum(bracket) % (len(bracket) - 2):
        raise ValueError("sum(bracket) must be divisible by len(bracket) - 2.")
    n = sum(bracket) // (len(bracket) - 2)
    offsets = [0]
    for value in bracket[1:]:
        offsets.append((offsets[-1] + n - value) % n)
    distinct = list(dict.fromkeys(offsets))
    if len(distinct) > len(ascii_lowercase):
        raise ValueError("Single-letter labels support at most 26 distinct reflections.")
    return dict(zip(distinct, ascii_lowercase))


def face_circuits_with_words(
    rotation: Rotation,
    adj: Rotation,
    n: int,
    reflection_labels: Dict[int, str],
) -> List[FaceCircuit]:
    """Trace each face once, covering each directed edge exactly once.

    A face may repeat vertices or undirected edges. Words are neither reduced
    nor deduplicated: different faces can have the same word. Each word starts
    with its earliest letter; ties use the lexicographically smallest cyclic
    rotation. Darts are shifted by the same amount to preserve their labels.
    """
    if set(rotation) != set(adj) or any(
        sorted(rotation[v]) != sorted(adj[v]) for v in adj
    ):
        raise ValueError("The rotation must list every vertex's neighbors exactly once.")
    sigma = make_sigma(rotation)
    alpha = make_alpha(adj)
    visited = set()
    faces = []
    for start in sorted(alpha):
        if start in visited:
            continue
        darts = []
        letters = []
        current = start
        while current not in visited:
            visited.add(current)
            darts.append(current)
            u, v = current
            r, s = (u, v - n) if u < n else (v, u - n)
            letters.append(reflection_labels[(s - r) % n])
            current = sigma[alpha[current]]
        if current != start:
            raise ValueError("Face traversal did not close at its starting dart.")
        word = "".join(letters)
        shift = min(range(len(word)), key=lambda i: word[i:] + word[:i])
        faces.append(FaceCircuit(
            darts=darts[shift:] + darts[:shift],
            word=word[shift:] + word[:shift],
        ))
    return faces


def run_face_word_experiment(
    bracket: List[int],
    max_iters: int = 2000000,
    init_temp: float = 3.0,
    final_temp: float = 1e-5,
    neighbor_func: Callable = neighbor_easy_swap,
    seed: int = None,
) -> ExperimentResult:
    """Run one annealing search, print its best result, and return the data.

    The final reported rotation is the best one returned by the existing
    annealer, not necessarily the last candidate visited or a proven optimum.
    A seed controls both the initial rotation and subsequent search moves.
    """
    if type(max_iters) is not int or max_iters < 0:
        raise ValueError("max_iters must be a nonnegative integer.")
    if any(not math.isfinite(t) or t <= 0 for t in (init_temp, final_temp)):
        raise ValueError("Temperatures must be finite and positive.")
    if final_temp > init_temp:
        raise ValueError("final_temp must not exceed init_temp for cooling.")
    reflection_labels = reflection_labels_from_bracket(bracket)
    adj = make_billiard_graph_integers(bracket)
    n = len(adj) // 2

    # The existing annealer's Euler formula assumes a connected graph.
    reached = {0}
    pending = [0]
    while pending:
        for neighbor in adj[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)
    if len(reached) != len(adj):
        raise ValueError("The bracket produces a disconnected graph; choose a connected bracket.")

    initial_rotation = random_rotation_from_adj(adj, seed=seed)
    started = time.perf_counter()
    rotation, genus = simulated_annealing_min_genus(
        adj, initial_rotation, neighbor_func,
        max_iters=max_iters, init_temp=init_temp, final_temp=final_temp,
    )
    elapsed = time.perf_counter() - started
    faces = face_circuits_with_words(rotation, adj, n, reflection_labels)
    result = ExperimentResult(adj, rotation, genus, reflection_labels, faces)

    print(f"\n=== Face Word Experiment: {bracket} ===")
    print(f"Neighbor method: {neighbor_func.__name__}")
    print(f"Iteration limit: {max_iters}; temperature: {init_temp} -> {final_temp}")
    print(f"Seed: {seed}; elapsed time: {elapsed:.2f}s")
    print(f"Best genus found: {genus}")
    print("\n=== Final Rotation (best found) ===")
    for vertex in sorted(rotation):
        print(f"vertex {vertex}: {rotation[vertex]}")
    print("\n=== Reflection Labels ===")
    print(f"n = {n}; label for edge r -- (n+s) is the letter for (s-r) mod n.")
    for offset, letter in reflection_labels.items():
        print(f"{letter}: reflection offset {offset} (mod {n})")
    print(f"\n=== Face/Circuit Words ({len(faces)} faces) ===")
    for index, face in enumerate(faces, start=1):
        print(f"Face {index} (length {len(face.darts)}): {face.word}")
        print("  Circuit: " + " -> ".join(map(str, face.vertices)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bracket", nargs="*", type=int, help="Bracket entries separated by spaces")
    parser.add_argument("--max-iters", type=int, default=2000000)
    parser.add_argument("--init-temp", type=float, default=3.0)
    parser.add_argument("--final-temp", type=float, default=1e-5)
    parser.add_argument("--neighbor", choices=NEIGHBOR_METHODS, default="easy_swap")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        bracket = args.bracket or [int(x) for x in input("Enter billiard bracket: ").split()]
        run_face_word_experiment(
            bracket, max_iters=args.max_iters,
            init_temp=args.init_temp, final_temp=args.final_temp,
            neighbor_func=NEIGHBOR_METHODS[args.neighbor], seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
