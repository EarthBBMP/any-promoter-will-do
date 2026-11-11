"""Within-sequence shuffles: in-distribution controls for the promoter branch.

Zeroing the promoter confounds two things, removing the promoter and handing the
network an input it could never have seen. These shuffles avoid that. Each one
returns a real DNA string of the same length and composition as the original,
so the model is never taken off-distribution; only the arrangement changes.

Two levels, because they destroy different things:

``mononucleotide``
    Permute bases uniformly. Base composition (including GC content) is
    preserved exactly; every dinucleotide, motif and positional relationship is
    destroyed.

``dinucleotide``
    Preserve the exact count of all 16 dinucleotides while randomising order,
    via the Altschul-Erikson Eulerian-path algorithm. This matters here because
    Xpresso reports CpG dinucleotide content at core promoters as strongly
    predictive of mRNA abundance: a mononucleotide shuffle would destroy CpG
    content and so confound "lost positional structure" with "lost CpG". The
    dinucleotide shuffle holds CpG fixed and removes only higher-order and
    positional structure.

Comparing the two therefore separates composition, dinucleotide content, and
positional arrangement as sources of the promoter branch's contribution.
"""

from typing import Dict, List

import numpy as np


def mononucleotide_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """Uniform permutation of the bases. Preserves composition exactly."""

    chars = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).copy()
    rng.shuffle(chars)

    return chars.tobytes().decode("ascii")


def _edge_lists(sequence: str) -> Dict[str, List[str]]:
    edges: Dict[str, List[str]] = {}
    for a, b in zip(sequence[:-1], sequence[1:]):
        edges.setdefault(a, []).append(b)

    return edges


def _reaches_last(last_edge: Dict[str, str], vertices, last: str) -> bool:
    """Do the designated last edges form a path to `last` from every vertex?"""

    for v in vertices:
        if v == last:
            continue
        seen = set()
        cur = v
        while cur != last:
            if cur in seen or cur not in last_edge:
                return False
            seen.add(cur)
            cur = last_edge[cur]

    return True


def dinucleotide_shuffle(sequence: str, rng: np.random.Generator) -> str:
    """
    Shuffle preserving exact dinucleotide counts (Altschul-Erikson).

    The sequence is an Eulerian path through the graph whose vertices are
    nucleotides and whose edges are the observed dinucleotides. Any other
    Eulerian path over the same edge multiset has identical dinucleotide counts,
    so we build one at random: designate for each vertex a final outgoing edge
    such that those edges form a path to the last nucleotide, shuffle the rest,
    then walk the graph.

    First and last nucleotides are preserved, as the algorithm requires.
    """

    if len(sequence) <= 2:
        return sequence

    edges = _edge_lists(sequence)
    first, last = sequence[0], sequence[-1]
    vertices = list(edges.keys())

    # Draw a valid arborescence of "last edges" pointing toward `last`.
    for _ in range(100):
        last_edge = {
            v: edges[v][rng.integers(len(edges[v]))] for v in vertices if v != last
        }
        if _reaches_last(last_edge, vertices, last):
            break
    else:  # pragma: no cover - vanishingly unlikely on real sequences
        return sequence

    # Shuffle every edge except the one reserved as each vertex's last.
    remaining: Dict[str, List[str]] = {}
    for v in vertices:
        pool = list(edges[v])
        if v in last_edge:
            pool.remove(last_edge[v])
        rng.shuffle(pool)
        if v in last_edge:
            pool.append(last_edge[v])
        remaining[v] = pool

    cursor = {v: 0 for v in vertices}
    out = [first]
    cur = first
    for _ in range(len(sequence) - 1):
        pool = remaining.get(cur)
        if pool is None or cursor[cur] >= len(pool):
            break
        nxt = pool[cursor[cur]]
        cursor[cur] += 1
        out.append(nxt)
        cur = nxt

    shuffled = "".join(out)

    # An incomplete walk would silently change composition; fall back rather
    # than return a sequence that is not a valid shuffle.
    return shuffled if len(shuffled) == len(sequence) else sequence


def dinucleotide_counts(sequence: str) -> Dict[str, int]:
    """Counts of all observed dinucleotides, for verifying a shuffle."""

    counts: Dict[str, int] = {}
    for a, b in zip(sequence[:-1], sequence[1:]):
        counts[a + b] = counts.get(a + b, 0) + 1

    return counts
