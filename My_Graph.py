class Node:
    def __init__(self, num):
        self.edges = set()
        self.super = self
        self.num = num
        self.val = 0

    def is_alive(self):
        return self.super == self

    def set_super(self,n_super):
        self.super = n_super

    def get_edges(self):
        return self.edges

    def dump_edges(self, to_dump):
        self.edges.difference_update(to_dump)

    def get_neighbors(self):
        neighbors = set()
        for edge in self.edges:
            ngh = edge.from_node if self is edge.to_node else edge.to_node
            neighbors.add(ngh)
        return neighbors

    def __str__(self):
        # TODO: CHANGE THIS!
        return str("N({})".format(str(self.num + 1)))

    def __repr__(self):
        return str(self)


class Blossom(Node):
    def __init__(self, num, cycle):
        super(Blossom, self).__init__(num)
        self.cycle = cycle

    def flower(self, exposed_dex, mate):
        if exposed_dex != 0:
            half1 = self.cycle[:exposed_dex]
            half2 = self.cycle[exposed_dex:]
            self.cycle = half2 + half1

        for i in range(1, len(self.cycle), 2):
            mate[self.cycle[i]] = self.cycle[i + 1]
            mate[self.cycle[i + 1]] = self.cycle[i]

        for ngh in self.get_neighbors():
            ngh.dump_edges(self.edges)




class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.super = self

    def is_alive(self):
        return self.super == self

    def set_super(self, n_super):
        self.super = n_super

    def wake(self):
        self.super = self

    def __str__(self):
        # TODO: CHANGE THIS!
        return "({}-({})-{})".format(self.from_node.num + 1, str(self.weight), self.to_node.num + 1)

    def __repr__(self):
        return str(self)


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.n = 0
        self.m = 0

    def add_nodes(self, m):
        n = len(self.nodes)
        new_nodes = [Node(i) for i in range(n, n+m)]
        self.nodes += new_nodes
        return new_nodes

    def get_node(self, num):
        return self.nodes[num]

    def add_edge(self, from_node, to_node, weight, update=True):
        new_edge = Edge(from_node, to_node, weight)
        from_node.edges.add(new_edge)
        to_node.edges.add(new_edge)
        if update:
            # dont want to save dummy edges
            self.edges.append(new_edge)
        return new_edge

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
