from collections import defaultdict, deque
import itertools
from itertools import permutations

import numpy as np
from tqdm import tqdm


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


"""
Complete the labelling of node using the 1-left-then-right rule
"""


def label_nodes(node, labels):
    """
    Given a list of labels inserts, them in the nodes of a tree.
    NOTE: trees are built from N leaves, there are N-1 nodes.
    """
    if not node.is_leaf:
        node.label = labels[0]
        labels = labels[1:]
        labels = label_nodes(node.left, labels)
        labels = label_nodes(node.right, labels)
    return labels


def far_right_insert(root, label):
    if root.is_leaf:
        root.label = label
    else:
        far_right_insert(root.right, label)


def label_leaves(root, rule):
    rule(root.left, root.label)
    if not root.left.is_leaf:
        label_leaves(root.left, rule)
    if not root.right.is_leaf:
        label_leaves(root.right, rule)


def make_tree(skeleton, labels):
    tree = build(skeleton)
    label_nodes(tree, labels)
    label_leaves(tree, far_right_insert)
    return tree


def get_permutation(root: Node | Leaf, sequence: str = "0"):
    """
    Extract the quasi-Stirling permutation from a fully labelled tree
    """
    sequence = sequence + root.label
    if type(root) == Node:
        sequence = get_permutation(root.left, sequence)
        sequence = get_permutation(root.right, sequence)
    return sequence


def make_all_labelled(n):
    skeletons = list(all_possible_trees(n))
    all_labels = list(map(str, range(1, n)))
    for skeleton in skeletons:
        for labels in permutations(all_labels):
            yield make_tree(skeleton, labels)


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


# %%

if __name__ == "__main__":
    # %%

    n = 8
    trees = list(all_possible_trees(n))
    ys = []
    for i, tree in enumerate(trees):
        root = build(tree)
        print(f"{i:>4}:", root)

    # %%

    # candidata tree as example
    tree = trees[8]

    # turn the list reprentations into nodes/leaves for easier/readable manipulation
    root = build(tree)

    print("Unlabelled tree:")
    print("", root)

    labels = list(map(str, range(1, n)))
    print("Labels:", labels)

    print("Partially labelled tree:")
    label_nodes(root, labels)

    print("", root)

    label_leaves(root, far_right_insert)

    print("Fully labelled tree:")
    print("", root)

    print("Pemruatation:")
    perm = get_permutation(root)
    print("", perm)

    # %%

    for i, tree in tqdm(enumerate(make_all_labelled(5))):
        print(f"{i:>4}: {get_permutation(tree)}")

    # %%

    ## TEST TREE
    # tree = build([[0, [[0, 0], 0]], 0])
    # labels = ["2", "3", "1", "4"]
    # label_tree(tree, labels)
    # apply_rule(tree, far_right_insert)
    # print(tree)
    # perm = get_permutation(tree)
    # print(perm)

    n = 8
    for i, tree in tqdm(enumerate(make_all_labelled(n))):
        permutation = get_permutation(tree)
        reconstructed_tree = perm_to_tree(permutation)
        reconstructed_perm = get_permutation(reconstructed_tree)
        if permutation != reconstructed_perm:
            print(f"ERROR:{permutation} != {reconstructed_perm}")
        # else:
        #     print(f"{permutation} MATCHES")

    # %%

    def descents_and_plateaus(
        perm,
    ):
        d, p = 0, 0
        for a, b in zip(perm, perm[1:]):
            # print(f"{a}{b}", end=" ")
            if a == b:
                # print("p")
                p += 1
            elif a > b:
                # print("d")
                d += 1
            # else:
            #     print("a")
        return d, p

    def descents_and_plateaus2(perm):
        d, p = 0, 0
        seen = set()
        for a, b in zip(perm, perm[1:]):
            # print(f"{a}{b}", end=" ")
            if a == b:
                p += 1
                # print("p1")
            elif a > b and a not in seen:
                p += 1
                d += 1
                # print("p2")
            elif a > b:
                d += 1
                # print("d")
            # else:
            #     print("a")
            seen.add(a)
        return d, p

    perm = "0432112340"
    print(perm)
    print(descents_and_plateaus(perm))
    print(descents_and_plateaus2(perm))

    # %%

    def make_dp(n):
        M = np.zeros((n, n), dtype=np.uint32)
        for tree in tqdm(make_all_labelled(n)):
            perm = get_permutation(tree)
            d, p = descents_and_plateaus(perm)
            M[p, d] += 1
        return M

    M = make_dp(7)

    print(M[1:, 1:], M.sum())

    # %%

    def is_stirling(perm):
        """
        Assume: perm is generated by a visit of a binary tree.
        The permutation is Stirling if:
            it has NO X_Y_Y_X : X < Y
        """
        perm = [int(i) for i in perm if int(i) > 0]
        ranges = defaultdict(list)
        for i, val in enumerate(perm):
            ranges[val].append(i)
        for val, (start, stop) in ranges.items():
            if np.min(perm[start : stop + 1]) < val:
                return False
        return True

    assert not is_stirling("0432112340")
    assert is_stirling("0123443210")
    assert is_stirling("0112233440")

    # %%

    # # %%
    #
    # from itertools import permutations
    #
    # def labels_skeleton_to_perm(labels, skeleton):
    #     tree = make_tree(skeleton, labels)
    #     return get_permutation(tree)
    #
    # PINK = "\033[95m"
    # BLUE = "\033[94m"
    # CYAN = "\033[96m"
    # GREEN = "\033[92m"
    # YELLOW = "\033[93m"
    # RED = "\033[91m"
    # ENDC = "\033[0m"
    # # BOLD = "\033[1m"
    # # UNDERLINE = "\033[4m"
    #
    # def color_char(ch, color):
    #     return f"{color}{ch}{ENDC}"
    #
    # def color_str(str, pos, color):
    #     return "".join(
    #         [color_char(ch, color) if i in pos else ch for i, ch in enumerate(str)]
    #     )
