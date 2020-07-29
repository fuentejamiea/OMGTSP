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

    def __str__(self):
        return "N({})".format(str(self.num))

    def __repr__(self):
        return str(self)


class Blossom(Node):
    def __init__(self, num, cycle, e_cycle, members):
        super(Blossom, self).__init__(num)
        self.cycle = cycle
        self.e_cycle = e_cycle
        self.members = members

    def is_blossom(self):
        return True

    def __str__(self):
        return "B({})".format(self.num)

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
        self.b_map = {}
        self.e_map = {}

    def add_nodes(self, m):
        n = len(self.nodes)
        new_nodes = [Node(i) for i in range(n, n+m)]
        self.nodes += new_nodes
        return new_nodes

    def add_edge(self, from_node, to_node, weight, update=True):
        new_edge = Edge(from_node, to_node, weight)
        from_node.edges.append(new_edge)
        to_node.edges.append(new_edge)
        self.edges.append(new_edge)
        return new_edge

    def add_blossom(self, cycle, e_cycle):
        n = len(self.nodes)
        new_blossom = Blossom(n, cycle, e_cycle, len(cycle))
        self.nodes.append(new_blossom)
        for node1 in cycle:
            self.b_map[node1] = new_blossom
            node1.kill()

        for node1 in cycle:
            if node1.is_blossom():
                new_blossom.members += node1.members - 1
            for edge in node1.edges:
                f_node = self.get_neighborhood(edge.from_node)
                t_node = self.get_neighborhood(edge.to_node)
                node2 = f_node if self.get_neighborhood(node1) is t_node else t_node
                if node2 is new_blossom:
                    self.e_map[edge] = new_blossom
                else:
                    new_blossom.edges.append(edge)

        return new_blossom

    def flower(self, blossom, mate):
        self.nodes.remove(blossom)

        if blossom in mate:
            b_mate = mate[blossom]
            ret = blossom.find_neighbor(b_mate)
            if not ret:
                print(blossom, b_mate)
                print(blossom.cycle)
                print([edge for edge in b_mate.edges if edge.active])
                print([edge for edge in b_mate.edges if not edge.active])
            i = ret[0]
            mate[b_mate] = blossom.cycle[i]
            mate.pop(blossom)
            half1 = blossom.cycle[:i]
            half2 = blossom.cycle[i:]
            blossom.cycle = half2 + half1
            mate[blossom.cycle[0]] = b_mate
        elif blossom.cycle[0] in mate:
            mate.pop(blossom.cycle[0])

        for k in range(1, len(blossom.cycle), 2):
            mate[blossom.cycle[k]] = blossom.cycle[k + 1]
            mate[blossom.cycle[k + 1]] = blossom.cycle[k]

        for node in blossom.cycle:
            self.b_map.pop(node)
            node.wake()
            for edge in node.edges:
                edge.wake()
        for edge in blossom.edges:
            neighbor = edge.to_node if edge.to_node is not blossom else edge.from_node
            neighbor.pop_edge(edge)

        return blossom

    def expand(self, mate):
        """
        :param mate:
            Matching return from maximal matching on Graph
            *mutates* expands blossoms to result in valid matching on original graph. b_nodes will be unmatched
        :return:
            None
        """
        b = []
        blossom = self.nodes[-1]
        while blossom.is_blossom():
            b.append(self.flower(blossom, mate))
            blossom = self.nodes[-1]
        return b

    def get_neighborhood(self, node):
        while node in self.b_map:
            node = self.b_map[node]
        return node

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
