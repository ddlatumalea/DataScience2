import networkx as nx
import matplotlib.pyplot as plt

def KhansAlgo(graph):
    """Returns topological ordering of a directed acyclic graph"""
    G = graph.copy()
    L = []
    S = [node for node in G.nodes if len(G.in_edges(node)) == 0]

    while S:
        n = S.pop(0)
        L.append(n)

        to_remove = []

        for edge in G.out_edges(n):
            to_remove.append(edge)

        for u, v in to_remove:
            G.remove_edge(u, v)
            if len(G.in_edges(nbunch=v)) == 0:
                S.append(v)

        to_remove.clear()

    if G.edges:
        return TypeError('Expects a directed acyclic graph.')

    return L

if __name__ == '__main__':
    # instantiate directional grapph
    G = nx.DiGraph()

    G.add_nodes_from([5, 7, 3, 11, 8, 2, 9, 10])
    G.add_edges_from([(5, 11), (7, 11), (7, 8), (3, 8), (3, 10),
                      (11, 2), (11, 9), (11, 10), (8, 9)])

    topological_ordering = KhansAlgo(G)
    print(topological_ordering)