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


def shrink_blossom(graph, node1, node2, parent):
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

    b1.reverse()
    cycle = b1 + b2
    b_node, neighbors = graph.add_blossom(cycle)

    for node in neighbors:
        if node in parent and parent[node] and not parent[node].is_alive():
            parent[node] = b_node

    parent[b_node] = parent[stem]
    return b_node



def aps(graph, mate, root):
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
    inner = set()
    outer = {root}
    if root.is_blossom():
        outer.update(root.cycle)
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
                    partner = mate[node2]
                    parent[partner] = node2
                    inner.add(node2)
                    if node2.is_blossom():
                        inner.update(node2.cycle)
                    outer.add(partner)
                    if partner.is_blossom():
                        outer.update(partner.cycle)
                    q.append(partner)

                elif node2 in outer:
                    # blossom detected
                    b_node = shrink_blossom(graph, node1, node2, parent)
                    outer.add(b_node)
                    outer.update(b_node.cycle)
                    q.appendleft(b_node)
                    if b_node.cycle[0] in mate:
                        mate[b_node] = mate[b_node.cycle[0]]
                        mate[b_node.cycle[0]] = b_node
                    if not node1.is_alive():
                        break
    return inner, outer


def maximal_matching(graph, mate=None, expand=True):
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
    inner = set()
    for root in graph.get_nodes():
        if root.is_alive() and root not in mate:
            # start augmenting path search (BFS format)
            o = aps(graph, mate, root)
            if o:
                inner.update(o[0])
                outer.update(o[1])
        if expand:
            graph.expand(mate)

    inner.difference_update(outer)
    return mate, inner, outer

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

    for node in graph.get_nodes():
        min_edge = min(node.edges, key=lambda e: e.weight)
        node.val = .5 * min_edge.weight

    n = len(graph.nodes)
    i = 0
    mate = {}

    while len(mate) < n:
        for edge in graph.edges:
            n1 = edge.to_node
            n2 = edge.from_node
            if n1.val + n2.val == edge.weight:
                edge.active = True
                if not n1.is_alive() or not n2.is_alive():
                    n1 = graph.get_neighborhood(n1)
                    n2 = graph.get_neighborhood(n2)
                    edge = graph.get_edge(n1, n2)
                    edge.active = True
            else:
                edge.active = False

        mate, inner, outer = maximal_matching(graph, mate=mate, expand=False)
        delta1 = delta2 = delta3 = float('inf')

        for e in graph.edges:
            n1 = e.to_node
            n2 = e.from_node
            n1_out = n1 in outer
            n1_in = n1 in inner
            n2_out = n2 in outer
            n2_in = n2 in inner
            if n1_out and n2_out and graph.get_neighborhood(n1) is not graph.get_neighborhood(n2):
                #print("delta1 update")
                #print(n1,n2, (e.weight - n1.val - n2.val)/2)
                delta1 = min(delta1, (e.weight - n1.val - n2.val)/2)
            elif (n1_out and not (n2_in or n2_out)) or (n2_out and not (n1_in or n1_out)):
                #print("delta2 update")
                #print(n1, n2, e.weight - n1.val - n2.val)
                delta2 = min(delta2, e.weight - n1.val - n2.val)

        for node in inner:
            if node.is_blossom():
                delta3 = min(delta3, -node.val/2)
        i += 1
        print("############round:{}###########".format(str(i)))
        print(sum([node.val for node in graph.get_nodes()]), [(node, node.val) for node in graph.get_nodes()])
        print([edge for edge in graph.get_edges() if edge.is_alive()])
        print(len(mate), mate)
        print("inner:", inner)
        print("outer:", outer)
        print(delta1, delta2, delta3)
        delta = min(delta1, delta2, delta3)
        for node in inner:
            if node.is_blossom():
                node.val += 2 * delta
                if node.val >= 0:
                    graph.flower(node, mate, True)
            else:
                node.val -= delta

        for node in outer:
            if node.is_blossom():
                node.val -= 2 * delta
                if node.val >= 0:
                    graph.flower(node, mate)
            else:
                node.val += delta
    return mate, sum([node.val for node in graph.get_nodes()])





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

wm = My_Graph.write_graph("Tests/weighted_matching.txt")
weighted_matching(wm)






