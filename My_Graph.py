class Node:
    def __init__(self, num):
        self.edges = set()
        self.num = num
        self.val = 0
        self.alive = True

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False

    def wake(self):
        self.alive = True

    def is_blossom(self):
        return False

    def get_edges(self):
        return list(self.edges)

    def pop_edge(self, e):
        self.edges.remove(e)

    def get_neighbors(self):
        neighbors = set()
        for edge in self.edges:
            if edge.is_alive():
                ngh = edge.from_node if self is edge.to_node else edge.to_node
                neighbors.add(ngh)
        return neighbors

    def __str__(self):
        return "N({})".format(str(self.num))

    def __repr__(self):
        # TODO: CHANGE THIS!
        return "N({})".format(str(self.num + 1))


class Blossom(Node):
    def __init__(self, num, cycle):
        super(Blossom, self).__init__(num)
        self.cycle = cycle

    def is_blossom(self):
        return True

    def __str__(self):
        return "B({})".format(str(self.num))

    def __repr__(self):
        return str(self)


class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.alive = True

    def is_alive(self):
        return self.alive

    def kill(self):
        self.alive = False

    def wake(self):
        self.alive = True

    def __str__(self):
        return "({}-({})-{})".format(self.from_node.num, str(self.weight), self.to_node.num)

    def __repr__(self):
        # TODO: CHANGE THIS!
        return "({}-({})-{})".format(self.from_node.num + 1, str(self.weight), self.to_node.num + 1)


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_nodes(self, m):
        n = len(self.nodes)
        new_nodes = [Node(i) for i in range(n, n+m)]
        self.nodes += new_nodes
        return new_nodes

    def add_edge(self, from_node, to_node, weight, update=True):
        new_edge = Edge(from_node, to_node, weight)
        from_node.edges.add(new_edge)
        to_node.edges.add(new_edge)
        if update:
            # dont want to save dummy edges
            self.edges.append(new_edge)
        return new_edge

    def add_blossom(self, cycle):
        n = len(self.nodes)
        new_blossom = Blossom(n, cycle)
        self.nodes.append(new_blossom)
        b_neighbors = {}
        for node in cycle:
            node.kill()
            self.nodes[node.num] = new_blossom

        for node1 in cycle:
            for edge in node1.edges:
                edge.kill()
                node2 = edge.from_node if node1 is edge.to_node else edge.to_node
                if node2.is_alive() and node2 not in b_neighbors:
                    self.add_edge(new_blossom, node2, -1, False)

        return new_blossom, b_neighbors

    def flower(self, blossom, stem_dex):
        for edge in blossom.edges:
            neighbor = edge.to_node
            neighbor.pop_edge(edge)

        for node in blossom.cycle:
            node.wake()
            self.nodes[node.num] = node
            for edge in node.edges:
                edge.wake()

        self.nodes[blossom.num] = self.nodes[-1]
        self.nodes.pop()

        if stem_dex != 0:
            half1 = blossom.cycle[:stem_dex]
            half2 = blossom.cycle[stem_dex:]
            return half2 + half1
        else:
            return blossom.cycle

    def get_node(self, num):
        if num < len(self.nodes):
            return self.nodes[num]
        else:
            return None

    def get_edge(self, from_node, to_node):
        return from_node.edges.intersection(to_node.edges)

    def get_edges(self):
        return self.edges




def write_graph(pathname):
    """
    :param pathname:
        Pathname to tsp txt file. Row 0 -> num_nodes num_edges, row 1: num_edges -> node1 node2 weight
    :return:
        Graph object representing TSP instance
    """
    fp = open(pathname)
    graph = Graph()
    graph.n, graph.m = map(int, fp.readline().split())
    node_list = graph.add_nodes(graph.n)

    for i in range(graph.m):
        n1, n2, weight = map(int, fp.readline().split())
        graph.add_edge(node_list[n1], node_list[n2], weight)
    fp.close()
    return graph
