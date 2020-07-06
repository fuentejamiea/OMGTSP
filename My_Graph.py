class Node:
    def __init__(self, num):
        self.num = num
        self.val = 0
        self.edges = []
        self.alive = True

    def is_alive(self):
        return self.alive

    def get_edges(self):
        return self.edges

    def __str__(self):
        return str("N({})".format(str(self.num)))

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

    def __str__(self):
        return "({}-({})-{})".format(self.from_node.num, str(self.weight), self.to_node.num)

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
