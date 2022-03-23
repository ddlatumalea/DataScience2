import networkx as nx
from collections import deque


def BFS(G, root, target=None):
    """Expects undirected acyclic connected graph"""
    Q = deque()
    explored = []

    Q.append(root)
    explored.append(root)

    # path traversal
    parents = dict()
    parents[root] = None

    path_found = False

    while Q:
        v = Q.popleft()

        print(v, end=" ")
        if target is not None and v == target:
            path_found = True
            break

        for w in G[v].keys():
            if w not in explored:
                explored.append(w)
                Q.append(w)
                parents[w] = v

    # traverse path
    path = []
    current_target = target
    if path_found:
        path.append(current_target)
        while parents[current_target] is not None:
            path.append(parents[current_target])
            current_target = parents[current_target]

    return path[::-1]


if __name__ == "__main__":
    G = nx.Graph()

    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    G.add_edges_from([(1, 2), (1, 3), (1, 4),
                      (2, 5), (2, 6),
                      (4, 7), (4, 8),
                      (5, 9), (5, 10),
                      (7, 11), (7, 12)])

    BFS(G, 1)

    # Another example
    G = nx.Graph()
    G.add_nodes_from([3, 5, 8, 25, 1, 2, 12, 8, 6, 4, 9])
    G.add_edges_from([
        (3, 5), (3, 8), (3, 25),
        (5, 1), (5, 2),
        (25, 12), (25, 8),
        (12, 6),
        (6, 4), (6, 9)
    ])

    print()
    print(BFS(G, 3, 4))