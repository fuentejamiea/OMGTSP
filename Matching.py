from collections import deque
import My_Graph


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
    b_neighbors = {}

    for node1 in b:
        node1.set_super(b_node)
        for edge in node1.edges:
            if edge.is_alive():
                node2 = edge.from_node if node1 is edge.to_node else edge.to_node
                if node2 not in cycle:
                    if node2 not in b_neighbors:
                        new_edge = graph.add_edge(b_node, node2, 0)
                        b_neighbors[node2] = new_edge
                        edge.set_super(new_edge)
                        if node2 in parent and parent[node2] in cycle:
                            parent[node2] = b_node
                    else:
                        edge.set_super(b_neighbors[node2])

                else:
                    edge.set_super(b_node)

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
    outer = {root}
    parent = {root: None}
    q = deque([root])
    while q:
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
                    return None

                elif node2 not in parent:
                    parent[node2] = node1
                    parent[mate[node2]] = node2
                    outer.add(mate[node2])
                    q.append(mate[node2])

                elif node2 in outer:
                    # blossom detected
                    b_node, cycle = blossom(graph, node1, node2, parent)
                    outer.add(b_node)
                    outer.update(cycle)
                    if cycle[-1] in mate:
                        mate[b_node] = mate[cycle[-1]]
                        mate[mate[cycle[-1]]] = b_node
                    b_list.append((b_node, cycle))
                    q.appendleft(b_node)
    return outer


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
    outer = set()
    b_list = []
    for root in graph.nodes:
        if root.is_alive() and root not in mate:
            # start augmenting path search (BFS format)
            o = aps(graph, mate, b_list, root)
            if o is not None:
                outer.update(o)

    return mate, b_list, outer


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
        node.set_super(node)

    graph.edges = graph.edges[:graph.m]
    for edge in graph.edges:
        edge.set_super(edge)


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

def weighted_matching(graph):

    b_list = []
    n = graph.n
    m = graph.m
    for node in graph.get_nodes():
        min_edge = min(node.edges, key=lambda e: e.weight)
        node.val = .5 * min_edge.weight

    for i in range(5):
        for edge in graph.edges[:m]:
            n1 = edge.to_node
            n2 = edge.from_node
            if n1.val + n2.val == edge.weight:
                edge.set_super(edge)
            else:
                edge.set_super(None)

        print([node.val for node in graph.get_nodes()])
        print([edge for edge in graph.get_edges() if edge.is_alive()])
        mate, b_update, outer = maximal_matching(graph)

        if b_update:
            b_list += b_update

        for b_info in b_list:
            if b_info[0] in outer:
                # add nodes in outer pseudonode to outer
                outer.update(b_info[1])

        inner = set()
        for node in outer:
            if node in mate and mate[node] not in outer:
                inner.add(mate[node])
        print(outer)
        print(inner)

        delta1 = delta2 = delta3 = float('inf')
        for e in graph.edges[:m]:
            if not isinstance(e.super, My_Graph.Node):
                n1 = e.to_node
                n2 = e.from_node
                n1_out = n1 in outer
                n1_in = n1 in inner
                n2_out = n2 in outer
                n2_in = n2 in inner
                if n1_out and n2_out:
                    #print("delta1 update")
                    #print(n1,n2, (e.weight - n1.val - n2.val)/2)
                    delta1 = min(delta1, (e.weight - n1.val - n2.val)/2)
                elif (n1_out and not (n2_in or n2_out)) or (n2_out and not (n1_in or n1_out)):
                    #print("delta2 update")
                    #print(n1, n2, e.weight - n1.val - n2.val)
                    delta2 = min(delta2, e.weight - n1.val - n2.val)

        for pseudo in graph.nodes[n:]:
            if pseudo in inner:
                delta3 = min(-pseudo.val/2)

        print(delta1,delta2,delta3)
        delta = min(delta1, delta2, delta3)
        for node in inner:
            if node.num < n:
                node.val -= delta
            else:
                node.val += 2*delta

        for node in outer:
            if node.num < n:
                node.val += delta
            else:
                node.val -= 2*delta

        print("##########################################")








def christofides(g):
    g = My_Graph.write_graph("TSP/ulysses22.txt")
    tree = min_spanning_tree(g)
    count = [False] * len(g.nodes)
    for e in tree:
        count[e.from_node.num] = not count[e.from_node.num]
        count[e.to_node.num] = not count[e.to_node.num]

    odd_nodes = {i for i in range(len(count)) if count[i]}
    weight_dict = {node: {} for node in odd_nodes}
    for i in odd_nodes:
        for j in odd_nodes:
            if j != i:
                e = g.get_edge(g.nodes[i], g.nodes[j])
                if e is not None:
                    weight_dict[i][j] = e.weight
                    weight_dict[j][i] = e.weight
                else:
                    weight_dict[i][j] = g.edges[-1].weight + 1
                    weight_dict[j][i] = g.edges[-1].weight + 1

    return weight_dict

#wm = My_Graph.write_graph("TSP/weighted_matching.txt")
#weighted_matching(wm)
