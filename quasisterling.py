import itertools
from math import ceil
from matplotlib import pyplot as plt

# A note on Stirling permutations                https://arxiv.org/pdf/2005.04133.pdf
# Descents on quasi-Stirling permutations        https://arxiv.org/pdf/2002.00985.pdf
# Pattern restricted quasi-Stirling permutations https://arxiv.org/pdf/1804.07267.pdf


def all_possible_trees(n):
    if n == 1:
        yield 0
    for split in range(1, n):
        gen_left = all_possible_trees(split)
        gen_right = all_possible_trees(n - split)
        for left, right in itertools.product(gen_left, gen_right):
            yield [left, right]


class Leaf:
    __slots__ = ("depth", "label", "is_leaf")

    def __init__(self, depth):
        self.depth: int = depth
        self.label: str = "0"
        self.is_leaf: bool = True

    def __repr__(self):
        return str(self.label)


class Node:
    __slots__ = ("depth", "label", "left", "right", "is_leaf")

    def __init__(self, left, right, depth):
        self.depth: int = depth
        self.label: str = "x"
        self.left: Node | Leaf = left
        self.right: Node | Leaf = right
        self.is_leaf: bool = False

    def __repr__(self):
        return f"[{self.label},{self.left},{self.right}]"


def build(root, depth=0):
    if root == 0:
        return Leaf(depth)
    left, right = root
    return Node(build(left, depth + 1), build(right, depth + 1), depth)


def test_trees():
    # compare against oies catalan numbers
    import requests
    import json

    url = "https://oeis.org/search?fmt=json&q=A000108&start=0"
    r = requests.get(url)
    if r.status_code == 200:
        response = json.loads(r.text)
        raw = response["results"][0]["data"]
        catalan = list(map(int, raw.split(",")))

        my_values = [len(list(all_possible_trees(n))) for n in range(1, 14)]
        for my, oies in zip(my_values, catalan):
            assert my == oies
            print(my)


# %%

n = 8
trees = list(all_possible_trees(n))
ys = []
for i, tree in enumerate(trees):
    root = build(tree)
    print(f"{i:0>8b}:", root)


# from math import factorial
## catalan number
# factorial(2 * (n - 1)) / (factorial((n - 1) + 1) * factorial(n - 1))


ntrees = len(trees)
print(ntrees)

# %%

# candidata tree as example
tree = trees[8]

# turn the list reprentations into nodes/leaves for easier/readable manipulation
root = build(tree)

print("Unlabelled tree:")
print(root)

labels = list(map(str, range(1, n)))
print("Labels:", labels)


def label_tree(node, labels):
    """
    Given a list of labels inserts, them in the nodes of a tree.
    NOTE: trees are built from N leaves, there are N-1 nodes.
    """
    if not node.is_leaf:
        node.label = labels[0]
        labels = labels[1:]
        labels = label_tree(node.left, labels)
        labels = label_tree(node.right, labels)
    return labels


print("Partially labelled tree:")
label_tree(root, labels)

print(root)

# %%

"""
Complete the labelling of node using the 1-left-then-right rule
"""


def far_right_insert(root, label):
    if root.is_leaf:
        root.label = label
    else:
        far_right_insert(root.right, label)


def apply_rule(root, rule):
    rule(root.left, root.label)
    if not root.left.is_leaf:
        apply_rule(root.left, rule)
    if not root.right.is_leaf:
        apply_rule(root.right, rule)


apply_rule(root, far_right_insert)

print("Fully labelled tree:", root)


# %%

"""
Extract the quasi-Stirling permutation from a fully labelled tree
"""


def get_permutation(root: Node | Leaf, sequence: str = "0"):
    sequence = sequence + root.label
    if type(root) == Node:
        sequence = get_permutation(root.left, sequence)
        sequence = get_permutation(root.right, sequence)
    return sequence


perm = get_permutation(root)


# %%

from collections import namedtuple

Edge = namedtuple("Edge", "parent_label child_label parent_y child_y parent_x child_x")


def visit(root):
    edges = []

    def _visit(root, depth=1, off=1.0):
        if root.is_leaf:
            return
        loff = off - 1 / 2**depth  # x offset of left child
        roff = off + 1 / 2**depth  # x offset of right child
        cdepth = -depth - 1  # depth of children (negative to draw root at top)
        edges.append(Edge(root.label, root.left.label, -depth, cdepth, off, loff))
        edges.append(Edge(root.label, root.right.label, -depth, cdepth, off, roff))
        _visit(root.left, depth=depth + 1, off=loff)
        _visit(root.right, depth=depth + 1, off=roff)

    _visit(root)

    return edges


def line(x0, y0, x1, y1, ax=None, color="black"):
    if ax:
        ax.plot((x0, x1), (y0, y1), color=color)
    else:
        plt.plot((x0, x1), (y0, y1), color=color)


def point(x0, y0, marker=".", ax=None):
    if ax:
        ax.plot((x0), (y0), marker, zorder=999)
    else:
        plt.plot((x0), (y0), marker, zorder=999)


def node(lbl, x0, y0, ax=None, **kwargs):
    if ax:
        ax.annotate(lbl, xy=(x0, y0), **kwargs)
    else:
        plt.annotate(lbl, xy=(x0, y0), **kwargs)


def get_top_node(edges):
    best = edges[0]
    for candidate in edges[1:]:
        if candidate.parent_y > best.parent_y:
            best = candidate
    return (best.parent_label, best.parent_x, best.parent_y)


def plot_single(tree, figsize=(8, 8)):
    _, ax = plt.subplots(figsize=figsize)

    plt.axis("off")

    edges = visit(tree)

    perm = get_permutation(tree)

    # centerd circle for nodes
    kwargs = {"ha": "center", "va": "center", "bbox": dict(boxstyle="circle", fc="1.0")}

    # fing top node to add stem
    label, x, y = get_top_node(edges)

    # draw stem above top node
    line(x0=x, x1=x, y0=y + 1, y1=y, ax=ax)
    node(lbl="0", x0=x, y0=y + 1, ax=ax, **kwargs)
    # each edge draws the destination node, the top node needs to be drawn separately
    node(lbl=label, x0=x, y0=y, ax=ax, **kwargs)

    for edge in edges:
        line(
            x0=edge.parent_x, x1=edge.child_x, y0=edge.parent_y, y1=edge.child_y, ax=ax
        )
        ## source node
        # node(edge.parent_label, x0=edge.parent_x, y0=edge.parent_y, ax=ax, **kwargs)
        ## destination node
        node(edge.child_label, x0=edge.child_x, y0=edge.child_y, ax=ax, **kwargs)

    plt.title(perm)
    plt.grid()
    plt.tight_layout()
    perm = get_permutation(tree)
    # plt.savefig(f"tree_figs/{perm}.png")
    # plt.close()
    plt.show()


def plot_all(trees, show_nodes=True, figsize=(16, 10)):
    ntrees = len(trees)
    N = ceil(ntrees**0.5)

    # if it fits nicely on a grid go for it
    if (N * (N - 1)) == ntrees:
        M = N - 1
    else:
        M = N

    _, axs = plt.subplots(nrows=N, ncols=M, figsize=figsize)

    for ax in axs.flatten():
        ax.axis("off")

    for ax, tree in zip(axs.flatten(), trees):
        root = build(tree)
        edges = visit(root)

        _, x, y = get_top_node(edges)
        point(x0=x, y0=y, ax=ax)

        for edge in edges:
            line(
                x0=edge.parent_x,
                x1=edge.child_x,
                y0=edge.parent_y,
                y1=edge.child_y,
                ax=ax,
            )
            if show_nodes:
                point(x0=edge.child_x, y0=edge.child_y, ax=ax)
            ax.grid()
    plt.tight_layout()
    plt.show()


# plot_all(list(all_possible_trees(7)))

# %%

n = 5
skeletons = list(all_possible_trees(n))

# n-1 strin labels
labels = list(map(str, range(1, n)))

from itertools import permutations
from tqdm import tqdm

for perm in tqdm(permutations(labels)):
    for skeleton in skeletons:
        tree = build(skeleton)
        label_tree(tree, perm)
        apply_rule(tree, far_right_insert)
        print(get_permutation(tree))
        # plot_single(tree)

# %%

from itertools import permutations
from string import ascii_uppercase


def labels_skeleton_to_perm(candidate, skeleton):
    tree = build(skeleton)
    label_tree(tree, candidate)
    apply_rule(tree, far_right_insert)
    return get_permutation(tree)


def is_stirling(perm):
    """
    Has no
        X_Y_Y_X : X < Y
    """
    seen = set()
    N = len(perm) // 2
    for start, el in enumerate(perm):
        seen.add(el)
        end = perm[start + 1 :].find(el) + start + 1
        for pos, val in enumerate(perm[start + 1 : end]):
            if int(el) > int(val):
                return False, (pos + start + 1, start, end)
        if len(seen) >= N:
            return True, None


# %%

## TEST TREE
# tree = build([[0, [[0, 0], 0]], 0])
# labels = ["2", "3", "1", "4"]
# label_tree(tree, labels)
# apply_rule(tree, far_right_insert)
# print(tree)
# perm = get_permutation(tree)
# print(perm)

from collections import deque


def perm_to_tree(perm):
    root = Node([], [], 0)
    root.label = perm[1]
    seen = set(perm[:2])

    current = root
    queue = deque()

    for a, b in zip(perm[1:], perm[2:]):
        # print("-", root, current, seen)
        if a == b:
            # add leaf to current
            # print(a, b, f"plateau, add leaf, label={b}")
            leaf = Leaf(current.depth + 1)
            leaf.label = b

            # NOTE: can this happen on the right?
            current.left = leaf

        if a != b:
            if b in seen:
                # print(a, b, f"second occurrence of b, label={b}")
                # add leaf to current
                leaf = Leaf(current.depth + 1)
                leaf.label = b

                current.right = leaf

                # pop pointer from queue
                while len(queue) > 0 and current.right != []:
                    # print("pop", current)
                    current = queue.pop()

            else:
                # print(a, b, "add subtree", end=" ")
                # add subtree
                next = Node([], [], current.depth + 1)
                next.label = b

                if current.left == []:
                    # print("to the left")
                    current.left = next
                else:
                    # print("to the right")
                    current.right = next

                # move pointer
                queue.append(current)
                current = next

                seen.add(b)
    return root


n = 8
skeletons = list(all_possible_trees(n))
labels = list(map(str, range(1, n)))
for perm in tqdm(permutations(labels)):
    for skeleton in skeletons:
        tree = build(skeleton)
        label_tree(tree, perm)
        apply_rule(tree, far_right_insert)
        permutation = get_permutation(tree)
        reconstructed_tree = perm_to_tree(permutation)
        reconstructed_perm = get_permutation(reconstructed_tree)
        if permutation != reconstructed_perm:
            print(f"ERROR:{permutation} != {reconstructed_perm}")
        # else:
        #     print(f"{permutation} MATCHES")


# %%

PINK = "\033[95m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
ENDC = "\033[0m"
# BOLD = "\033[1m"
# UNDERLINE = "\033[4m"


def color_char(ch, color):
    return f"{color}{ch}{ENDC}"


def color_str(str, pos, color):
    return "".join(
        [color_char(ch, color) if i in pos else ch for i, ch in enumerate(str)]
    )


n = 5

# as if xD
assert n < 26

skeletons = list(all_possible_trees(n))
labels = list(map(str, range(1, n)))  # n-1 labels

candidates = list(permutations(labels))
for skeleton in skeletons:
    perms = []
    pattern = labels_skeleton_to_perm(candidates[0], skeleton)
    pattern = "".join(map(lambda x: ascii_uppercase[int(x)], pattern))

    print("Skeleton:", pattern)
    for candidate in candidates:
        perm = labels_skeleton_to_perm(candidate, skeleton)
        perms.append(perm)
        res, pos = is_stirling(perm)
        c = "".join(candidate)
        if res:
            print(c, " -> ", perm, "S")
        else:
            perm_augmented = color_str(perm, pos, RED)
            values = set([int(perm[p]) for p in pos])
            c_pos = [c.find(str(v)) for v in values]
            c_augmented = color_str(c, c_pos, RED)
            print(c_augmented, " -> ", perm_augmented, "")
    print()


# %%

import numpy as np

n = 7

skeletons = list(all_possible_trees(n))
labels = list(map(str, range(1, n)))  # n-1 labels

rows = []
candidates = list(permutations(labels))
for skeleton in skeletons:
    perms = []
    row = []
    for candidate in candidates:
        perm = labels_skeleton_to_perm(candidate, skeleton)
        perms.append(perm)
        row.append(is_stirling(perm)[0])
    rows.append(row)

mat = np.array(rows)

# %%

to_show = np.flipud(mat)
spaces = np.zeros_like(to_show)
to_show = np.hstack([to_show, spaces]).reshape(to_show.shape[0] * 2, to_show.shape[1])
print(to_show.shape)

# N = to_show.shape[0]
# to_show = np.array([np.roll(row, (N - i) // 2) for i, row in enumerate(to_show)])

plt.figure(figsize=(19, 8))
# plt.axis("off")
plt.ylabel("skeletons")
plt.xlabel("permutations")
plt.imshow(to_show, interpolation="nearest")
plt.tight_layout()
plt.show()
