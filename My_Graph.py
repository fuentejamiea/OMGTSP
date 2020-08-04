class Node:
    def __init__(self, num):
        self.edges = []
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

    def get_edge(self, node):
        """
        :param node:
            other node in edge
        :return:
            edge (self, node) if exists, else None
        """
        for e in self.edges:
            if node is e.from_node or node is e.to_node:
                return e

    def pop_edge(self, e):
        """
        :param e:
            edge to be removed from edge list
        :return:
            None, edits node's edge list
        """
        i = self.edges.index(e)
        self.edges[i] = self.edges[-1]
        self.edges.pop()

    def __le__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented

    def __lt__(self, other):
        return self.num < other.num

    def __gt__(self, other):
        return self.num > other.num

    def __str__(self):
        return "N({})".format(str(self.num))

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
        return "({}-({})-{})".format(str(self.from_node), str(self.weight), str(self.to_node))

    def __repr__(self):
        return str(self)


class Graph:
    def __init__(self, pathname=""):
        """
        :param pathname:
            Pathname to tsp txt file. Row 0 -> num_nodes num_edges, row 1: num_edges -> node1 node2 weight
        """

        self.nodes = []
        self.edges = []
        if pathname:
            fp = open(pathname)
            n, m = map(int, fp.readline().split())
            self.add_nodes(n)

            for i in range(m):
                n1, n2, weight = map(int, fp.readline().split())
                self.add_edge(self.nodes[n1], self.nodes[n2], weight)
            fp.close()

    def add_nodes(self, m):
        """
        :param m:
            number of nodes to be added
        :return:
            None
        """
        n = len(self.nodes)
        for i in range(n, n+m):
            self.nodes.append(Node(i))

    def add_edge(self, from_node, to_node, weight):
        """
        :param from_node:
            node object representing tail of edge (graph does not currently support directed edges)
        :param to_node:
            node object representing tip of edge (graph does not currently support directed edges)
        :param weight: e
            dge weight
        :return:
            new edge object
        """
        new_edge = Edge(from_node, to_node, weight)
        from_node.edges.append(new_edge)
        to_node.edges.append(new_edge)
        self.edges.append(new_edge)
        return new_edge

    def get_node(self, num):
        if num < len(self.nodes):
            return self.nodes[num]
        else:
            return None

    def get_edge(self, from_node, to_node):
        if len(from_node.edges) < len(to_node.edges):
            return from_node.get_edge(to_node)
        else:
            return to_node.get_edge(from_node)

    def get_nodes(self):
        return list(self.nodes)

    def get_edges(self):
        return list(self.edges)
