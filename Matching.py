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
    print("backtracking: \n")
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
    print(n1)
    print(e1)

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
    print(n2)
    print(e2)

    if n1[-1] == n2[-1]:
        #blossom detected
        print("shrinking \n")
        i = -2
        while n1[i] != n2[i]:
            i -= 1

        n1 = n1[:i]
        e1 = e1[:i]
        n2 = n2[:i - 1]
        e2 = e2[:i]
        print(n1, e1)
        print(n2, e2)

        n1.reverse()
        e1.reverse()
        cycle = n1 + n2
        e_cycle = e1 + e2
        print(cycle, e_cycle)
        b_node = graph.add_blossom(cycle, e_cycle)
        print("\n new blossom:\n")
        print(b_node)
        print(b_node.edges)
        print(b_node.members)
        outer[b_node] = outer[cycle[0]]
        if cycle[0] in mate:
            mate[b_node] = mate[cycle[0]]
            mate.pop(cycle[0])
        print("done \n")
        return b_node
    print("augmenting\n")
    n1.reverse()
    e1.reverse()
    p = n1 + n2
    e = e1 + e2
    print(p)
    print(e)
    for i in range(0, len(e), 2):
        mate[p[i]] = e[i]
        mate[p[i + 1]] = e[i]
    print("****************\n")
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
            print(node1.edges)
            for edge in node1.edges:
                e_flag = edge is not mate[node1] if node1 in mate else True
                if edge.is_alive() and e_flag:
                    f_node = graph.get_neighborhood(edge.from_node)
                    t_node = graph.get_neighborhood(edge.to_node)
                    node2 = f_node if node1 is t_node else t_node
                    if node2 in outer:
                        print("outer")
                        print(outer)
                        print(inner)
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
                print("inner")
                print(outer)
                print(inner)
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
    if mate is None:
        mate = {}

    outer_n = set()
    outer_b = set()
    inner_n = set()
    inner_b = set()
    maximal = False
    while not maximal:
        roots = [node for node in graph.nodes if node.is_alive() and node not in mate]
        print(mate)
        print(roots)
        print("####################")
        maximal = aps(graph, mate, roots)
        if expand:
            graph.flower(mate)

    #blossom_set_update(inner_n, inner_b, maximal[0])
    #blossom_set_update(outer_n, outer_b, maximal[1])
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

    print([(nodes, nodes.val) for nodes in graph.nodes])

    n = len(graph.nodes)
    mate = {}
    k = 0
    delta = 1

    while delta != float('inf'):
        print("############round:{}###########".format(str(k)))
        k += 1
        for edge in graph.edges:
            n1 = edge.to_node
            n2 = edge.from_node
            s_val = []
            neigh1 = graph.get_neighborhood_list(n1)
            neigh2 = graph.get_neighborhood_list(n2)
            i = -1
            b1 = neigh1[i]
            b2 = neigh2[i]
            if b1 is b2:
                while b1 is b2:
                    s_val.append(b1)
                    i -= 1
                    b1 = neigh1[i]
                    b2 = neigh2[i]

            if n1.val + n2.val + sum(b.val for b in s_val) == edge.weight:
                edge.active = True

                if not n1.is_alive() or not n2.is_alive():
                    edge = graph.get_edge(neigh2[-1], neigh1[-1])
                    edge.active = True

            else:
                edge.active = False
        mate, inner_n, inner_b, outer_n, outer_b = maximal_matching(graph, mate=mate, expand=False)
        if len(mate) == n:
            break

        print(len(mate), mate)
        print("left:", len(outer_n) - len(inner_n) - 2 * sum([(len(b.members) -1) /2 for b in outer_b]) +
              2 * sum([(len(b.members) -1) /2 for b in inner_b]))
        print("inner_b", inner_b)
        print("outer_b", outer_b)
        print("inner_n",inner_n)
        print("outer_n", outer_n)
        print([(b, b.cycle) for b in graph.nodes[n:]])
        delta1 = delta2 = delta3 = float('inf')

        for e in graph.edges:
            n1 = e.to_node
            n2 = e.from_node
            n1_out = n1 in outer_n
            n1_in = n1 in inner_n
            n2_out = n2 in outer_n
            n2_in = n2 in inner_n
            if n1_out and n2_out and graph.get_neighborhood(n1) is not graph.get_neighborhood(n2):
                if (e.weight - n1.val - n2.val)/2 <= 0:
                    print("d1", n1,n2)
                delta1 = min(delta1, (e.weight - n1.val - n2.val)/2)
            elif (n1_out and not (n2_in or n2_out)) or (n2_out and not (n1_in or n1_out)):
                if (e.weight - n1.val - n2.val) <= 0:
                    print("d2", n1,n2)
                delta2 = min(delta2, e.weight - n1.val - n2.val)

        for node in inner_b:
            if node.val != 0:
                delta3 = min(delta3, -node.val/2)
        delta = min(delta1, delta2, delta3)
        print(delta1, delta2, delta3)
        if delta <= 0:
            return -1

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

            if blossom.val == 0 and blossom.is_alive():
                graph.flower(blossom, mate)
            else:
                j -= 1
            blossom = graph.nodes[j]
        print([node.val for node in graph.nodes])
        s1 = sum([nodes.val for nodes in graph.nodes[:n]])
        s2 = sum([b.val * ((len(b.members) - 1)/2) for b in graph.nodes[n:]])
        print(s1 + s2)

    return s1 + s2


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
