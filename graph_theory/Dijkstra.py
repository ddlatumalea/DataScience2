import networkx as nx
import numpy as np


def Dijkstra(graph, start_node, end_node):
    """
    Returns a tuple containing the total distances from the start node and the vertices on the path.

    :param graph: a networkx graph
    :param start_node: Int starting node
    :param end_node: Int node to find a path to
    """
    paths = {v: np.inf for v in graph.nodes}
    prev_nodes = {v: np.inf for v in graph.nodes}

    paths[start_node] = 0
    prev_nodes[start_node] = 0

    nodes_visited = [start_node]

    while end_node not in nodes_visited:
        for node in nodes_visited:
            neighbours = graph[node]
            for neighbour, props in neighbours.items():
                if paths[node] + props['distance'] < paths[neighbour]:
                    paths[neighbour] = paths[node] + props['distance']
                    prev_nodes[neighbour] = node

        valid_nodes = {k: v for k, v in paths.items() if k not in nodes_visited}
        node_min = min(valid_nodes, key=valid_nodes.get)

        nodes_visited.append(node_min)

    # Returning the traversed path
    traversed_path = []
    current_node = end_node

    traversed_path.append(current_node)

    while start_node not in traversed_path:
        current_node = prev_nodes[current_node]
        traversed_path.append(current_node)

    path_distance = [paths[node] for node in traversed_path]

    return path_distance[::-1], traversed_path[::-1]


if __name__ == '__main__':
    vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    edges = [
        (0, 2, {'distance': 3}),
        (0, 3, {'distance': 1}),
        (0, 4, {'distance': 6}),
        (1, 2, {'distance': 2}),
        (1, 3, {'distance': 7}),
        (2, 0, {'distance': 3}),
        (2, 1, {'distance': 2}),
        (2, 5, {'distance': 1}),
        (4, 5, {'distance': 5}),
        (4, 6, {'distance': 3}),
        (5, 2, {'distance': 1}),
        (5, 4, {'distance': 5}),
        (5, 7, {'distance': 4}),
        (6, 3, {'distance': 4}),
        (6, 4, {'distance': 3}),
        (6, 7, {'distance': 2}),
        (7, 6, {'distance': 2}),
        (7, 5, {'distance': 4})
    ]

    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    shortest_path = Dijkstra(G, 0, 5)
    print(shortest_path)
