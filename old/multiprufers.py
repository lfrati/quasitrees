import json
from itertools import permutations

from trees import (
    all_prufers,
    edges2tree,
    extract_sequences,
    get_multitree,
    multivisit,
    prufer2edges,
    visit,
)

#%%


def test():

    forest = {
        1: set([2, 3]),
        2: set({1, 4}),
        3: set([1, 5, 6, 7]),
        4: set([2]),
        5: set([3]),
        6: set([3]),
        7: set([3, 8]),
        8: set([7]),
    }

    seq = visit(forest, 1, "pre")
    print("".join(map(str, seq)))
    seq = visit(forest, 1, "post")
    print("".join(map(str, seq)))
    seq = visit(forest, 1, "both")
    print("".join(map(str, seq)))
    print()

    perms = list(extract_sequences(forest, "pre"))
    for p in sorted(perms):
        print(p)
    print("Tot:", len(perms), "Unique:", len(set(perms)))

    print(get_multitree(forest, 3))
    print()

    for seq in multivisit(forest, 3, "pre"):
        print("".join(map(str, seq)))

    print()

    for seq in multivisit(forest, 1, "pre"):
        print("".join(map(str, seq)))

    print()

    for seq in multivisit(forest, 4, "pre"):
        print("".join(map(str, seq)))


def main():
    # test()

    for N in range(3, 8):
        print("Computing N = ", N)
        codes = all_prufers(N)
        codes2perms = {}
        for code in codes:
            edges = prufer2edges(code)
            forest = edges2tree(edges)
            perms = list(extract_sequences(forest, "both"))
            codes2perms["".join(map(str, code))] = sorted(perms)

        all_perms = [el for val in codes2perms.values() for el in val]
        n_perms = len(all_perms)
        print("Tot:", len(all_perms))
        assert len(all_perms) == len(list(set(all_perms)))

        with open(f"prufers_N={N}_{n_perms}.json", "w") as f:
            f.write(json.dumps(codes2perms))


if __name__ == "__main__":
    main()
