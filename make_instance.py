from collections import deque
from My_Graph import Graph
import numpy as np
import pickle
import gurobipy


def write_graph(pathname):
    """
    :param pathname:
        Pathname to tsp txt file. Row 0 -> num_nodes num_edges, row 1: num_edges -> node1 node2 weight
    :return:
        Graph object representing TSP instance
    """
    fp = open(pathname)
    graph = Graph()
    num_nodes, num_edges = map(int, fp.readline().split())
    graph.add_nodes(num_nodes)

    for i in range(num_edges):
        v1, v2, weight = map(int, fp.readline().split())
        node1 = graph.get_node(v1)
        node2 = graph.get_node(v2)
        graph.add_edge(node1, node2, weight)
    fp.close()
    graph.order_edges()
    return graph


def min_spanning_tree(graph):
    """
    :param graph:
       My_Graph graph object with edges sorted by weight
    :return:
        Set of My_Graph edge objects representing MST
    """
    n = len(graph.nodes)
    num_edges = 0
    tree = set()
    seen = [False] * n
    for e in graph.edges:
        if not seen[e.from_node.num] or not seen[e.to_node.num]:
            seen[e.from_node.num] = True
            seen[e.to_node.num] = True
            tree.add(e)
            num_edges += 1
        if num_edges == n - 1:
            break

    return tree


def blossom(graph, node1, node2, parent):
    """
    :param graph:
        Graph object from My_Graph Module
        *mutated* Blossom shrunk and replaced by Node b_node
    :param node1:
        Node1 in blossom (int)
    :param node2:
        Node2 in blossom (int)
    :param parent:
        parent[Node] = Node's BFS tree Parent, parent[root] = None
        *mutated* Node s.t. parent[Node] in blossom -> parent[Node] = b_node
    :return:
        b_node = Node object representing blossom node
        b = List of blossom Nodes in APS cycle order
    """
    cycle = set()
    b1 = []
    b2 = []
    start1, start2 = node1, node2
    while True:
        if node1 is not None:
            if node1 in cycle:
                stem = node1
                break
            cycle.add(node1)
            node1 = parent[node1]

        if node2 is not None:
            if node2 in cycle:
                stem = node2
                break
            cycle.add(node2)
            node2 = parent[node2]

    while start1 is not parent[stem]:
        b1.append(start1)
        start1 = parent[start1]

    while start2 is not stem:
        b2.append(start2)
        start2 = parent[start2]

    b2.reverse()
    b = b2 + b1
    cycle = set(b)
    b_node = graph.add_nodes(1)[0]
    parent[b_node] = parent[b[-1]]
    b_neighbors = set()

    for node1 in b:
        node1.alive = False
        for edge in node1.edges:
            if edge.is_alive():
                edge.alive = False
                node2 = edge.from_node if node1 is edge.to_node else edge.to_node
                if node2 not in cycle:
                    b_neighbors.add(node2)

    for neighbor in b_neighbors:
        graph.add_edge(b_node, neighbor, 0)
        if neighbor in parent and parent[neighbor] in cycle:
            parent[neighbor] = b_node

    return b_node, b


def aps(graph, mate, b_list, root):
    """
    :param graph:
        Graph object from My_Graph module
        *mutated* shrinks blossoms and adds blossom nodes and edges
    :param mate:
        Dict s.t. mate[node1] = node1's mate in current matching
        *mutated* augmented when augmenting path/blossom is found
    :param b_list:
        List of (b_node, blossom) pairs. blossom is a list of Nodes
        *mutated* (b_node, blossom) pairs added to list as found
    :param root:
        Node that starts augmenting path search
    :return:
        none

    """
    visited = set()
    basis = {root}
    parent = {root: None}
    q = deque([root])
    cont = True
    while cont and q:
        node1 = q.popleft()
        if not node1.is_alive() or node1 in visited:
            continue
        visited.add(node1)
        for edge in node1.edges:
            if edge.is_alive():
                node2 = edge.from_node if node1 is edge.to_node else edge.to_node
                if node2 not in mate and node2 not in parent:
                    # found augmenting path
                    parent[node2] = node1
                    while node2 is not None:
                        mate[node2] = parent[node2]
                        mate[parent[node2]] = node2
                        node2 = parent[parent[node2]]
                    cont = False
                    break

                elif node2 not in parent:
                    parent[node2] = node1
                    parent[mate[node2]] = node2
                    basis.add(mate[node2])
                    q.append(mate[node2])

                elif node2 in basis:
                    # blossom detected
                    b_node, cycle = blossom(graph, node1, node2, parent)
                    basis.add(b_node)
                    if cycle[-1] in mate:
                        mate[b_node] = mate[cycle[-1]]
                        mate[mate[cycle[-1]]] = b_node
                    b_list.append((b_node, cycle))
                    q.appendleft(b_node)


def maximal_matching(graph, mate=None):
    """
    :param graph:
        Graph Object from My_Graph module
        *mutated* see aps docstring
    :param mate:
        (optional) initial matching
        *mutated* updated to matching of higher cardinality if one exists
    :return:
        mate: Maximal cardinality matching on Graph by Blossom algorithm
        b_list: list of (b_node, blossom) pairs. blossom is a list of Nodes
    """
    if mate is None:
        mate = {}
    b_list = []
    for root in graph.nodes:
        if root.is_alive() and root not in mate:
            # start augmenting path search (BFS format)
            aps(graph, mate, b_list, root)

    return mate, b_list


def flower(graph, mate, b_list):
    """
    :param graph:
        My_Graph Graph object after maximal matching has been called
        *mutates* expands blossoms and deletes blossom nodes and edges
    :param mate:
        Matching return from maximal matching on Graph
        *mutates* expands blossoms to result in valid matching on original graph. b_nodes will be unmatched
    :param b_list:
        List of (b_node, blossom) pairs where blossom is a list of nodes in blossom
    :return:
        None
    """
    for i in range(len(b_list) - 1, -1, -1):
        b_node, cycle = b_list[i]
        if b_node in mate and mate[b_node] is not None:
            b_mate = mate[b_node]
            for j in range(len(cycle)):
                e = graph.get_edge(b_mate, cycle[j])
                if e is not None and not e.is_alive():
                    half1 = cycle[:j]
                    half2 = cycle[j:]
                    cycle = half2 + half1
                    mate[b_mate] = cycle[0]
                    mate[cycle[0]] = b_mate
                    break
        else:
            mate[cycle[0]] = None

        mate[b_node] = None
        cycle[0].alive = True
        for k in range(1, len(cycle), 2):
            mate[cycle[k]] = cycle[k + 1]
            mate[cycle[k + 1]] = cycle[k]

    graph.nodes = graph.nodes[:graph.n]
    for node in graph.nodes:
        node.edges = node.edges[:node.val]
        node.alive = True

    graph.edges = graph.edges[:graph.m]
    for edge in graph.edges:
        edge.alive = True


def clean_matching(mate, valid):
    """
    :param mate:
        Matching in M[node] = mate format
    :param valid:
        Function to determine if Node is valid for given matching
    :return:
        Matching in {(node1,node2), (node3,node4)... ] format
    """
    matching = set()
    for node1, node2 in mate.items():
        if valid(node1) and valid(node2):
            if node2.num < node1.num:
                matching.add((node1.num, node2.num))
            else:
                matching.add((node2.num, node1.num))
    return matching


def random_matching_test(n, num, denom, pickle_path=False):
    """
    :param n:
        number of Nodes in test Graph
    :param num:
        numerator in num/denom = probability there exists an edge between any two distinct nodes
    :param denom:
        denominator in num/denom = probability there exists an edge between any two distinct nodes
    :param pickle_path:
        (optional) path to pickled random mat representing node adj. matrix
    :return:
        diff = difference of cardinality of maximal matching algorithm and cardinality of matching found by gurobi ILP
        invalid = number edges in matching from algo not in graph
    """
    cutoff = denom - num
    g1 = Graph()
    model = gurobipy.Model()
    model.setParam("OutputFlag", False)
    g1.add_nodes(n)
    g1.n = n
    con_map = {i: set() for i in range(n)}
    if pickle_path:
        with open(pickle_path, 'rb') as f:
            mat = pickle.load(f)
    else:
        mat = []

    for i in range(n):
        if not pickle_path:
            new_row = np.random.randint(0, denom, i)
            mat.append(new_row)
        for j in range(i):
            if mat[i][j] >= cutoff:
                n1 = g1.get_node(i)
                n2 = g1.get_node(j)
                n1.val += 1
                n2.val += 1
                g1.add_edge(n1, n2, 0)
                g1.m += 1
                new_var = model.addVar(obj=-1, vtype=gurobipy.GRB.BINARY, name="({},{})".format(i, j))
                model.update()
                con_map[i].add(new_var)
                con_map[j].add(new_var)

    for nd, const in con_map.items():
        model.addConstr(gurobipy.quicksum(const) <= 1)
    model.update()

    mate, b = maximal_matching(g1)
    flower(g1, mate, b)
    matching = clean_matching(mate, lambda node: node is not None)
    count1 = len(matching)
    matching = {m for m in matching if mat[m[0]][m[1]] >= cutoff}
    count2 = len(matching)
    invalid = count1 - count2
    model.optimize()
    grb_match = [v.VarName for v in model.getVars() if v.x != 0]
    diff = len(matching) - len(grb_match)
    if diff != 0 or invalid != 0:
        with open('problem_mat.pkl', 'wb') as f:
            pickle.dump(mat, f)
    return diff, invalid
