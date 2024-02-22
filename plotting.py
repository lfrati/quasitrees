from collections import namedtuple
from math import ceil

import matplotlib.pyplot as plt

from quasisterling import build, get_permutation, all_possible_trees

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


# %%
if __name__ == "__main__":
    #%%
    plot_all(list(all_possible_trees(7)))
