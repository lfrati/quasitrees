from numpy.random import choice
from itertools import permutations, product
from collections import defaultdict


def prufer2edges(prufer):
    # wiki https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence
    # from https://www.geeksforgeeks.org/prufer-code-tree-creation/
    # alternative code: https://github.com/stevenschmatz/prufer

    vertices = len(prufer) + 2
    # Initialize the array of vertices
    vertex_set = [0] * vertices
    # Number of occurrences of vertex in code
    for i in prufer:
        vertex_set[i - 1] += 1

    edges = []

    # Find the smallest label not present in prufer.
    for i in prufer:
        for j in range(vertices):
            # If j+1 is not present in prufer set
            if vertex_set[j] == 0:
                # Remove from Prufer set and add edge.
                vertex_set[j] = -1
                vertex_set[i - 1] -= 1
                edges.append(((j + 1), i))
                break

    # For the last element
    last = []
    j = 0
    for i in range(vertices):
        if vertex_set[i] == 0 and j == 0:
            last.append(i + 1)
            j += 1
        elif vertex_set[i] == 0 and j == 1:
            last.append(i + 1)

    edges.append(tuple(last))
    return edges


def random_prufer(l):
    # a prufer seq of length l corresponds to a labelled tree with l+2 nodes
    return choice(range(1, l + 1), size=l - 2)


def all_prufers(N):
    # generate all prufer seqs for labelled trees with N nodes
    opts = list(range(1, N + 1))
    prufers = list(product(opts, repeat=N - 2))
    # check https://en.wikipedia.org/wiki/Cayley%27s_formula
    assert len(prufers) == N ** (N - 2)
    return prufers


def visit(tree, root=0):
    seen = set()
    seq = ""
    return _visit(tree, root, seen, seq)


def _visit(tree, root, seen, seq):
    seen.add(root)
    seq += str(root)
    for node in tree[root]:
        if node not in seen:
            seq = _visit(tree, node, seen, seq)
    seq += str(root)
    return seq


def edges2tree(edges):
    tree = defaultdict(set)
    for (start, end) in edges:
        tree[start].add(end)
        tree[end].add(start)
    tree = dict(tree)
    return tree


# A note on Stirling permutations                https://arxiv.org/pdf/2005.04133.pdf
# Descents on quasi-Stirling permutations        https://arxiv.org/pdf/2002.00985.pdf
# Pattern restricted quasi-Stirling permutations https://arxiv.org/pdf/1804.07267.pdf


def unique_permutations(s):
    N = len(s)
    seen = set()
    for p in permutations(s, N):
        if p not in seen:
            seen.add(p)
            yield p
