from collections import deque, Counter
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


def backtrack(graph, edge, inner, outer, mate):
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
    n1 = []
    e1 = [edge]
    n2 = []
    e2 = []
    node = graph.get_neighborhood(edge.from_node)
    if node in inner:
        n1.append(node)
        e1.append(inner[node])
        f_node = graph.get_neighborhood(inner[node].from_node)
        t_node = graph.get_neighborhood(inner[node].to_node)
        node = f_node if node is t_node else t_node

    while outer[node]:
        n1.append(node)
        e1.append(outer[node])
        f_node = graph.get_neighborhood(outer[node].from_node)
        t_node = graph.get_neighborhood(outer[node].to_node)
        node = f_node if node is t_node else t_node
        n1.append(node)
        e1.append(inner[node])
        f_node = graph.get_neighborhood(inner[node].from_node)
        t_node = graph.get_neighborhood(inner[node].to_node)
        node = f_node if node is t_node else t_node
    n1.append(node)

    node = graph.get_neighborhood(edge.to_node)
    if node in inner:
        n2.append(node)
        e2.append(inner[node])
        f_node = graph.get_neighborhood(inner[node].from_node)
        t_node = graph.get_neighborhood(inner[node].to_node)
        node = f_node if node is t_node else t_node

    while outer[node]:
        n2.append(node)
        e2.append(outer[node])
        f_node = graph.get_neighborhood(outer[node].from_node)
        t_node = graph.get_neighborhood(outer[node].to_node)
        node = f_node if node is t_node else t_node
        n2.append(node)
        e2.append(inner[node])
        f_node = graph.get_neighborhood(inner[node].from_node)
        t_node = graph.get_neighborhood(inner[node].to_node)
        node = f_node if node is t_node else t_node
    n2.append(node)

    if n1[-1] is n2[-1]:
        i = 1
        l1 = len(n1)
        l2 = len(n2)
        while n1[l1 - i] is n2[l2 - i]:
            i += 1

        n1 = n1[:l1 - i + 2]
        e1 = e1[:l1 - i + 2]
        n2 = n2[:l2 - i + 1]
        e2 = e2[:l2 - i + 1]

        n1.reverse()
        e1.reverse()
        cycle = n1 + n2
        e_cycle = e1 + e2
        b_node = graph.add_blossom(cycle, e_cycle, inner)
        outer[b_node] = outer[cycle[0]]
        if cycle[0] in mate:
            mate[b_node] = mate[cycle[0]]
            mate.pop(cycle[0])
        return b_node
    n1.reverse()
    e1.reverse()
    p = n1 + n2
    e = e1 + e2
    for i in range(0, len(e), 2):
        mate[p[i]] = e[i]
        mate[p[i + 1]] = e[i]
    print("agu p:", p)
    return False


def aps(graph, mate, roots):
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
    inner = {}
    outer = {root: None for root in roots}
    q = deque(roots)
    while q:
        node1 = q.popleft()
        if not node1.is_alive() or node1 in visited:
            continue
        visited.add(node1)
        if node1 in outer:
            for edge in node1.edges:
                e_flag = edge is not mate[node1] if node1 in mate else True
                if edge.is_alive() and e_flag:
                    f_node = graph.get_neighborhood(edge.from_node)
                    t_node = graph.get_neighborhood(edge.to_node)
                    node2 = f_node if node1 is t_node else t_node
                    if node2 in outer:
                        blossom = backtrack(graph, edge, inner, outer, mate)
                        if not blossom:
                            return False
                        q.append(blossom)
                        break

                    elif node2 not in inner:
                        inner[node2] = edge
                        q.append(node2)

        if node1 in inner:
            edge = mate[node1]
            f_node = graph.get_neighborhood(edge.from_node)
            t_node = graph.get_neighborhood(edge.to_node)
            node2 = f_node if node1 is t_node else t_node
            if node2 in inner:
                blossom = backtrack(graph, edge, inner, outer, mate)
                if not blossom:
                    return False
                q.append(blossom)

            elif node2 not in outer:
                outer[node2] = edge
                q.append(node2)

    return inner, outer


def blossom_set_update(node_set, blossom_set, update):
    for node in update:
        if node.is_blossom():
            blossom_set_update(node_set, blossom_set, node.cycle)
            if node.is_alive():
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
    print("############matching###########")
    if mate is None:
        mate = {}
    n = len(graph.nodes)
    print("edges:", [e for e in graph.edges if e.is_alive()])

    maximal = False
    while not maximal:
        roots = [node for node in graph.nodes if node.is_alive() and node not in mate]
        print("mate:", set(mate.values()))
        print([(b, b.cycle) for b in graph.nodes[n:]])
        print("roots:", roots)
        maximal = aps(graph, mate, roots)
        if maximal:
            print("inner:", maximal[0])
            print("outer:", maximal[1])
        if expand:
            graph.flower(mate)

    return mate, maximal[0], maximal[1]


def weighted_matching(graph):

    for node in graph.get_nodes():
        min_edge = min(node.edges, key=lambda edge: edge.weight)
        node.val = .5 * min_edge.weight

    n = len(graph.nodes)
    mate = {}
    k = 0
    print("nodes:",[(n, n.val) for n in graph.nodes])

    while len(mate) < n:
        for edge in graph.edges:
            n1 = edge.to_node
            n2 = edge.from_node
            s_val = graph.get_edge_val(n1, n2)
            if n1.val + n2.val + s_val == edge.weight:
                edge.wake()
            else:
                edge.kill()

        mate, inner, outer = maximal_matching(graph, mate=mate, expand=False)
        #if len(mate) == n:
            #break

        outer_n = set()
        outer_b = set()
        inner_n = set()
        inner_b = set()
        blossom_set_update(inner_n, inner_b, inner)
        blossom_set_update(outer_n, outer_b, outer)

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
                delta2 = min(delta2, e.weight - n1.val - n2.val)

        for node in inner_b:
            delta3 = min(delta3, -node.val/2)
        delta = min(delta1, delta2, delta3)

        for node in inner_n:
            node.val -= delta

        for node in outer_n:
            node.val += delta
        print("############round:{}###########".format(str(k)))
        print("blossoms:",[(b, b.cycle) for b in graph.nodes[n:]])

        j = -1
        blossom = graph.nodes[j]
        while blossom.is_blossom():
            if blossom in outer_b:
                blossom.val -= 2 * delta
            elif blossom in inner_b:
                blossom.val += 2 * delta

            if blossom.val >= 0 and blossom.is_alive():
                graph.expand(blossom, mate)
            else:
                j -= 1
            blossom = graph.nodes[j]

        s1 = sum([nodes.val for nodes in graph.nodes[:n]])
        s2 = sum([b.val * ((b.members - 1)/2) for b in graph.nodes[n:]])
        k += 1
        print("blossoms:",[(b, b.cycle) for b in graph.nodes[n:]])
        print(len(mate), set(mate.values()))
        print("unmatched:",[n for n in graph.nodes if n not in mate])
        print("inner_b", inner_b)
        print("outer_b", outer_b)
        print("inner_n", inner_n)
        print("outer_n", outer_n)
        print(delta1,delta2,delta3)
        print([(node, node.val) for node in graph.nodes])
        print("obj:",s1 + s2)
        print("matching weight:", sum([e.weight for e in mate.values()])/2)
    print("***********expand*************")
    graph.flower(mate)

    print("matching weight:", sum([e.weight for e in mate.values()]) / 2)
    print(set(mate.values()))
    return s1 + s2


def christofides(g):
    g = My_Graph.write_graph("TSP/ulysses22.txt")
    g.order_edges()
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
