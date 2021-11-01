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


def edges2tree(edges):
    tree = defaultdict(set)
    for (start, end) in edges:
        tree[start].add(end)
        tree[end].add(start)
    tree = dict(tree)
    return tree


def visit(tree, root, order="both"):
    seen = set()
    seq = []
    return _visit(tree, root, seen, seq, order)


def _visit(tree, root, seen, seq, order="both"):
    seen.add(root)
    if order == "pre" or order == "both":
        seq.append(root)
    children = [node for node in tree[root] if node not in seen]
    for node in children:
        seq = _visit(tree, node, seen, list(seq), order)
    if order == "post" or order == "both":
        seq.append(root)
    return seq


def get_multitree(forest, root):
    # Turn
    # { 1: {2, 3},
    #   2: {1, 4},
    #   3: {1, 5, 6, 7},
    #   4: {2},
    #   5: {3},
    #   6: {3},
    #   7: {8, 3},
    #   8: {7}}
    # into
    # { 1: [(2, 3), (3, 2)],
    #   2: [(4,)],
    #   4: [()],
    #   3: [(5, 6, 7), (5, 7, 6), (6, 5, 7), (6, 7, 5), (7, 5, 6), (7, 6, 5)],
    #   5: [()],
    #   6: [()],
    #   7: [(8,)],
    #   8: [()]
    # }
    seen = set()
    options = {}
    return _options(forest, root, seen, options)


def _options(forest, root, seen, options):
    seen.add(root)
    children = [node for node in forest[root] if node not in seen]
    options[root] = list(permutations(children))
    for node in children:
        _options(forest, node, seen, options)
    return options


def instantiate(multitree):
    options = list(multitree.items())
    tree = {}
    yield from _instantiate(tree, options)


def _instantiate(tree, options):
    if len(options) == 0:
        # no options available, yield the current tree
        yield dict(tree)
    else:
        # options available, explore every combination
        node, values = options[0]
        for val in values:
            tree[node] = val
            yield from _instantiate(tree, options[1:])


def multivisit(forest, root, order="both"):
    # given a forest calculate possible visits order
    # then instantiate each one of them and return visit sequence
    multitree = get_multitree(forest, root)
    visits = [visit(t, root, order) for t in instantiate(multitree)]
    return visits


def extract_sequences(forest, order="both"):
    # return generator that explores a given tree from each possible root and each
    # visit order
    for root in forest.keys():
        seqs = multivisit(forest, root, order)
        # convert visit sequence to string e.g. [[1,2,2,1],[1,1,2,2]] -> ["1221","1122"]
        seqs = ["".join(map(str, seq)) for seq in seqs]
        yield from seqs


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
