from Heuristic import Matching
from Graph import My_Graph


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

    path = [graph.nodes[0]]

    circuit = []

    while path:

        cur_node = path[-1]

        if cur_node.edges:
            edge = cur_node.edges.pop()
            neighbor = edge.to_node if edge.from_node is cur_node else edge.from_node
            neighbor.edges.remove(edge)
            path.append(neighbor)
        else:
            circuit.append(path.pop())

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
            match.add_edge(node_num1, node_num2, edge.weight)

    mate, match_weight = match.weighted_matching()
    euler = My_Graph.Graph()
    euler.add_nodes(n)
    for edge in mate:
        euler.add_edge(match_to_graph[edge.from_node.num],
                       match_to_graph[edge.to_node.num], edge.weight)

    for edge in tree:
        node1 = euler.nodes[edge.from_node.num]
        node2 = euler.nodes[edge.to_node.num]
        euler.add_edge(node1, node2, edge.weight)

    tour = eulerian_tour(euler)
    node_count = {}
    for i in range(len(tour)):
        node = tour[i]
        if node in node_count:
            node_count[node].add(i)
        else:
            node_count[node] = {i}

    node_count = {n: c for n, c in node_count.items() if len(c) > 1}
    node_index = {}
    l = len(tour)
    for node in node_count:
        best_index = -1
        short_val = float('inf')
        for index in node_count[node]:
            e1 = euler.get_edge(tour[index - 1].num, node.num)
            next_dex = index + 1 if index + 1 < l else 0
            e2 = euler.get_edge(node.num, tour[next_dex].num)
            tour_weight = e1.weight + e2.weight
            shortcut = tour_weight - graph.get_edge(tour[index - 1].num, tour[next_dex].num).weight
            if shortcut < short_val:
                short_val = shortcut
                best_index = index

        node_index[node] = best_index

    ham_cycle = []
    for i in range(len(tour)):
        node = tour[i]
        if node in node_index:
            if node_index[node] == i:
                ham_cycle.append(node)
        else:
            ham_cycle.append(node)

    weight = 0
    for i in range(len(ham_cycle)):
        edge = graph.get_edge(ham_cycle[i - 1].num, ham_cycle[i].num)
        weight += edge.weight


    return ham_cycle, weight






















