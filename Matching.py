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
    b_list = []
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
                    for i in range(len(b_list) - 1, -1, -1):
                        graph.flower(b_list[i], mate)
                    return None

                elif node2 not in parent:
                    parent[node2] = node1
                    partner = mate[node2]
                    parent[partner] = node2
                    inner.add(node2)
                    outer.add(partner)
                    q.append(partner)

                elif node2 in outer:
                    # blossom detected
                    b_node = shrink_blossom(graph, node1, node2, parent)
                    b_list.append(b_node)
                    outer.add(b_node)
                    q.appendleft(b_node)
                    if b_node.cycle[0] in mate:
                        mate[b_node] = mate[b_node.cycle[0]]
                        mate[mate[b_node.cycle[0]]] = b_node
                        mate.pop(b_node.cycle[0])
                    if not node1.is_alive():
                        break
    return inner, outer

def blossom_set_update(node_set, blossom_set, update):
    for node in update:
        if node.is_blossom():
            blossom_set_update(node_set, blossom_set, node.cycle)
            blossom_set.add(node)
        else:
            node_set.add(node)


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
    outer_n= set()
    outer_b = set()
    inner_n = set()
    inner_b = set()
    for root in graph.get_nodes():
        if root.is_alive() and root not in mate:
            # start augmenting path search (BFS format)
            o = aps(graph, mate, root)
            if o:
                blossom_set_update(inner_n, inner_b, o[0])
                blossom_set_update(outer_n, outer_b, o[1])
        if expand:
            graph.expand(mate)

    inner_n.difference_update(outer_n)
    inner_b.difference_update(outer_b)
    return mate, inner_n, inner_b, outer_n, outer_b



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
            if node2.num > node1.num:
                matching.add((node2.num, node1.num))
            else:
                matching.add((node1.num, node2.num))
    return matching


def weighted_matching(graph):

    for node in graph.get_nodes():
        min_edge = min(node.edges, key=lambda e: e.weight)
        node.val = .5 * min_edge.weight

    n = len(graph.nodes)
    i = 1
    mate = {}

    while len(mate) < n:
        print("############round:{}###########".format(str(i)))
        active_edges = []
        for edge in graph.edges:
            n1 = edge.to_node
            n2 = edge.from_node
            bval = 0
            neigh1 = graph.get_neighborhood_list(n1)
            neigh2 = graph.get_neighborhood_list(n2)
            i = -1
            b1 = neigh1[i]
            b2 = neigh2[i]
            if b1 is b2:
                while b1 is b2:
                    i -= 1
                    b1 = neigh1[i]
                    b2 = neigh2[i]
                bval = neigh1[i + 1].val




            if n1.val + n2.val + bval == edge.weight:
                edge.active = True
                active_edges.append(edge)

                if not n1.is_alive() or not n2.is_alive():
                    for b in neigh1:
                        edge = graph.get_edge(b, b2)
                        if edge:
                            edge.active = True

                    for b in neigh2:
                        edge = graph.get_edge(b, b1)
                        if edge:
                            edge.active = True

            else:
                edge.active = False

        print([(b, b.cycle) for b in graph.nodes[n:]])

        mate, inner_n, inner_b, outer_n, outer_b = maximal_matching(graph, mate=mate, expand=False)
        delta1 = delta2 = delta3 = float('inf')

        for e in graph.edges:
            n1 = e.to_node
            n2 = e.from_node
            n1_out = n1 in outer_n
            n1_in = n1 in inner_n
            n2_out = n2 in outer_n
            n2_in = n2 in inner_n
            if n1_out and n2_out and graph.get_neighborhood(n1) is not graph.get_neighborhood(n2):
                delta1 = min(delta1, (e.weight - n1.val - n2.val)/2)

            elif (n1_out and not (n2_in or n2_out)) or (n2_out and not (n1_in or n1_out)):
                if e.weight - n1.val - n2.val <= 0:
                    print(n1, n2, n1.val, n2.val, e.weight)
                delta2 = min(delta2, e.weight - n1.val - n2.val)

        for node in inner_b:
            if node.val != 0:
                delta3 = min(delta3, -node.val/2)
        i += 1


        print(len(mate), mate)
        print("inner_b:", inner_b)
        print("inner_n:", inner_n)
        print("outer_b:", outer_b)
        print("outer_n:", outer_n)
        print(delta1, delta2, delta3)
        delta = min(delta1, delta2, delta3)

        if i > n + 1 or delta <= 0:
            print("#############################")
            return {}

        for node in inner_n:
            node.val -= delta

        for node in outer_n:
            node.val += delta

        j = -1
        blossom = graph.nodes[j]
        while blossom.is_blossom():
            if blossom in outer_b:
                blossom.val -= 2 * delta
            elif blossom in inner_b:
                blossom.val += 2 * delta

            if blossom.val >= 0:
                graph.flower(blossom, mate)
            else:
                j -= 1
            blossom = graph.nodes[j]
    print("#############################")
    graph.expand(mate)

    return mate


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
