class Node:
    def __init__(self, num):
        self.super = self
        self.num = num
        self.val = 0
        self.edges = []

    def is_alive(self):
        return self.super == self

    def set_super(self,n_super):
        self.super = n_super

    def get_edges(self):
        return self.edges

    def __str__(self):
        # TODO: CHANGE THIS!
        return str("N({})".format(str(self.num + 1)))

    def __repr__(self):
        return str(self)


class Edge:
    def __init__(self, from_node, to_node, weight):
        self.super = self
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    def is_alive(self):
        return self.super == self

    def set_super(self,n_super):
        self.super = n_super

    def __str__(self):
        # TODO: CHANGE THIS!
        return "({}-({})-{})".format(self.from_node.num + 1, str(self.weight), self.to_node.num + 1)

    def __repr__(self):
        return str(self)


class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.e_map = {}
        self.n = 0
        self.m = 0

    def add_nodes(self, m):
        n = len(self.nodes)
        new_nodes = [Node(i) for i in range(n, n+m)]
        if n == 0:
            self.nodes = new_nodes
        else:
            self.nodes += new_nodes
        return new_nodes

    def get_node(self, node_num):
        n = len(self.nodes)
        if node_num >= n:
            print("only {} nodes in graph".format(n))
            return None

        return self.nodes[node_num]

    def get_nodes(self):
        return self.nodes

    def add_edge(self, from_node, to_node, weight):
        if not isinstance(from_node, Node):
            to_node = self.get_node(to_node)

        if not isinstance(to_node, Node):
            from_node = self.get_node(from_node)

        new_edge = Edge(from_node, to_node, weight)
        from_node.edges.append(new_edge)
        to_node.edges.append(new_edge)
        self.edges.append(new_edge)
        return new_edge

    def get_edge(self, from_node, to_node):
        for edge in from_node.edges:
            if to_node is edge.from_node or to_node is edge.to_node:
                return edge
        return None

    def get_edges(self):
        return self.edges

    def order_edges(self):
        self.edges.sort(key=lambda e: e.weight)


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
    graph.add_nodes(graph.n)

    for i in range(graph.m):
        v1, v2, weight = map(int, fp.readline().split())
        node1 = graph.get_node(v1)
        node2 = graph.get_node(v2)
        graph.add_edge(node1, node2, weight)
    fp.close()
    graph.order_edges()
    return graph
