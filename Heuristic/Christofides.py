from Heuristic import Matching
from Graph import My_Graph
from collections import defaultdict


def min_spanning_tree(graph):
    """
    :param graph:
        My_Graph graph object
    :return:
        Min spanning tree using prims O(n^2) algorithm. Since TSP graph is complete no
        need to use binary heap
    """
    n = len(graph.nodes)
    if not n:
        return []
    tree = [My_Graph.Edge] * n
    distance = [float('inf')] * n
    in_tree = [False] * n

    cur_node = graph.nodes[0]
    for edge in cur_node.edges:
        neighbor = edge.to_node.num if edge.from_node is cur_node else edge.from_node.num
        tree[neighbor] = edge
        distance[neighbor] = edge.weight

    in_tree[0] = True
    distance[0] = 0
    for _ in range(n - 1):
        min_val = float('inf')
        for num in range(n):
            if (not in_tree[num]) and distance[num] < min_val:
                min_val = distance[num]
                cur_node = num

        in_tree[cur_node] = True
        for edge in graph.nodes[cur_node].edges:
            neighbor = edge.to_node.num if edge.from_node.num is cur_node else edge.from_node.num
            if (not in_tree[neighbor]) and edge.weight < distance[neighbor]:
                distance[neighbor] = edge.weight
                tree[neighbor] = edge

    return tree[1:], sum(distance)


def eulerian_tour(graph):
    """
    :param graph: 
        An even My_Graph graph object
    :return: 
        Eulerian tour via Hierholzer's algorithm
    """

    path = [(graph.nodes[0], None)]

    circuit = []

    while path:

        cur_node = path[-1][0]

        if cur_node.edges:
            edge = cur_node.edges.pop()
            neighbor = edge.to_node if edge.from_node is cur_node else edge.from_node
            neighbor.edges.remove(edge)
            path.append((neighbor, edge))
        else:
            circuit.append(path.pop()[1])

    for edge in graph.edges:
        edge.from_node.edges.add(edge)
        edge.to_node.edges.add(edge)

    circuit.pop()
    return circuit


def christofides(graph):
    """
    :param graph:
        Complete metric My_Graph graph object
    :return:
        Hamiltonian tour and weight
    """
    n = len(graph.nodes)
    if not n:
        return [], 0

    tree, tree_weight = min_spanning_tree(graph)
    odd_degree = [False] * n
    for edge in tree:
        odd_degree[edge.from_node.num] = not odd_degree[edge.from_node.num]
        odd_degree[edge.to_node.num] = not odd_degree[edge.to_node.num]

    num_odd = 0
    graph_to_match = {}
    match_to_graph = {}
    for i in range(n):
        if odd_degree[i]:
            match_to_graph[num_odd] = i
            graph_to_match[i] = num_odd
            num_odd += 1

    match = Matching.Matching()
    match.add_nodes(num_odd)
    for edge in graph.edges:
        if odd_degree[edge.from_node.num] and odd_degree[edge.to_node.num]:
            node_num1 = graph_to_match[edge.from_node.num]
            node_num2 = graph_to_match[edge.to_node.num]
            m_edge = match.add_edge(node_num1, node_num2, edge.weight)
            m_edge.num = edge.num

    mate, match_weight = match.weighted_matching()
    euler = My_Graph.Graph()
    euler.add_nodes(n)
    for edge in mate:
        e_edge = euler.add_edge(match_to_graph[edge.from_node.num],
                       match_to_graph[edge.to_node.num], edge.weight)
        e_edge.num = edge.num

    for edge in tree:
        node1 = euler.nodes[edge.from_node.num]
        node2 = euler.nodes[edge.to_node.num]
        e_edge = euler.add_edge(node1, node2, edge.weight)
        e_edge.num = edge.num

    tour = eulerian_tour(euler)
    ham_tour = []
    cur_node = euler.nodes[0]
    visited = set()
    i = 0
    while len(ham_tour) < n:
        next_node = tour[i].from_node if tour[i].to_node is cur_node else tour[i].to_node
        if next_node in visited:
            while next_node in visited:
                i += 1
                next_node = tour[i].from_node if tour[i].to_node is next_node else tour[i].to_node
            ham_tour.append(graph.get_edge(cur_node.num, next_node.num))
        else:
            ham_tour.append(graph.edges[tour[i].num])
        visited.add(next_node)
        cur_node = next_node
        i += 1

    return ham_tour
