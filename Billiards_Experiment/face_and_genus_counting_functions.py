import random

from typing import Dict, List, Tuple, Set

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

def face_length_counts(
    rotation: Dict[int, List[int]],
    adj: Dict[int, List[int]]
) -> Dict[int, int]:
    """
    Count how many faces have each length.

    Returns a dictionary where:
        key   = number of sides of the face
        value = number of faces of that size
    """

    sigma = make_sigma(rotation)
    alpha = make_alpha(adj)
    darts = build_darts(adj)

    visited: Set[Tuple[int, int]] = set()
    face_lengths: Dict[int, int] = {}

    for d in darts:
        if d in visited:
            continue

        cur = d
        face_length = 0

        while True:
            visited.add(cur)
            face_length += 1

            cur = sigma[alpha[cur]]

            if cur in visited:
                break

        # Add this face to the appropriate count
        if face_length not in face_lengths:
            face_lengths[face_length] = 0

        face_lengths[face_length] += 1

    return face_lengths

def print_face_length_counts(
    rotation: Dict[int, List[int]],
    adj: Dict[int, List[int]]
) -> None:
    """Print the number of faces of each size."""

    face_counts = face_length_counts(rotation, adj)

    print("\n=== Face Length Distribution ===")
    print(f"Total number of faces: {sum(face_counts.values())}")

    for length in sorted(face_counts):
        print(f"{length}-sided faces: {face_counts[length]}")


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