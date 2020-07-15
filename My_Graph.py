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
        for e in self.edges:
            if node is e.from_node or node is e.to_node:
                return e
        return None

    def pop_edge(self, e):
        i = self.edges.index(e)
        self.edges[i] = self.edges[-1]
        self.edges.pop()

    def get_neighbors(self):
        neighbors = set()
        for edge in self.edges:
            if edge.is_alive():
                ngh = edge.from_node if self is edge.to_node else edge.to_node
                neighbors.add(ngh)
        return neighbors

    def __str__(self):
        # TODO: CHANGE THIS!
        return "N({})".format(str(self.num))

    def __repr__(self):
        return str(self)


class Blossom(Node):
    def __init__(self, num, cycle):
        super(Blossom, self).__init__(num)
        self.cycle = cycle

    def is_blossom(self):
        return True

    def find_neighbor(self, node):
        for i in range(len(self.cycle)):
            candidate = self.cycle[i]
            e = candidate.get_edge(node)
            if e:
                return i, e
        return None

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
        return "({}-({})-{})".format(str(self.from_node), str(self.weight), str(self.to_node))

    def __repr__(self):
        return str(self)


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
        from_node.edges.append(new_edge)
        to_node.edges.append(new_edge)
        if update:
            # dont want to save dummy edges
            self.edges.append(new_edge)
        return new_edge

    def add_blossom(self, cycle):
        n = len(self.nodes)
        new_blossom = Blossom(n, cycle)
        self.nodes.append(new_blossom)
        neighbors = set()

        for node in cycle:
            node.kill()
            self.nodes[node.num] = new_blossom

        for node1 in cycle:
            for edge in node1.edges:
                edge.kill()
                node2 = edge.from_node if node1 is edge.to_node else edge.to_node
                if node2.is_alive() and node2 not in neighbors:
                    self.add_edge(new_blossom, node2, -1, False)
                    neighbors.add(node2)

        return new_blossom, neighbors

    def flower(self, blossom, mate):
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

        if blossom in mate and mate[blossom] is not None:
            b_mate = mate[blossom]
            i, _ = blossom.find_neighbor(b_mate)
            mate[b_mate] = blossom.cycle[i]
            mate.pop(blossom)
            half1 = blossom.cycle[:i]
            half2 = blossom.cycle[i:]
            blossom.cycle = half2 + half1
        else:
            b_mate = None

        mate[blossom.cycle[0]] = b_mate
        for k in range(1, len(blossom.cycle), 2):
            mate[blossom.cycle[k]] = blossom.cycle[k + 1]
            mate[blossom.cycle[k + 1]] = blossom.cycle[k]

    def expand(self, mate):
        """
        :param mate:
            Matching return from maximal matching on Graph
            *mutates* expands blossoms to result in valid matching on original graph. b_nodes will be unmatched
        :return:
            None
        """
        b_node = self.nodes[-1]
        while b_node.is_blossom():
            self.flower(b_node, mate)
            b_node = self.nodes[-1]

    def get_neighborhood(self, node):
        index = node.num
        while self.nodes[index].num != index:
            index = self.nodes[index].num
        return self.nodes[index]

    def get_node(self, num):
        if num < len(self.nodes):
            return self.nodes[num]
        else:
            return None

    def get_nodes(self):
        return list(self.nodes)

    def get_edges(self):
        return list(self.edges)


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
