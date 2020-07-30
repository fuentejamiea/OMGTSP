from collections import deque

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
        self.cycle = deque(cycle)
        self.e_cycle = deque(e_cycle)
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

    def add_blossom(self, cycle, e_cycle, inner):
        n = len(self.nodes)
        new_blossom = Blossom(n, cycle, e_cycle, len(cycle))
        self.nodes.append(new_blossom)
        for node1 in cycle:
            self.b_map[node1] = new_blossom
            inner.pop(node1, None)
            node1.kill()

        for node1 in cycle:
            if node1.is_blossom():
                new_blossom.members += node1.members - 1
            for edge in node1.edges:
                f_node = self.get_neighborhood(edge.from_node)
                t_node = self.get_neighborhood(edge.to_node)
                node2 = f_node if self.get_neighborhood(node1) is t_node else t_node
                if node2 is not new_blossom:
                    new_blossom.edges.append(edge)

        return new_blossom

    def flower(self, mate):
        blossom = self.nodes[-1]
        while blossom.is_blossom():
            self.expand(blossom, mate)
            blossom = self.nodes[-1]

    def expand(self, blossom, mate):
        """
        :param mate:
            Matching return from maximal matching on Graph
            *mutates* expands blossoms to result in valid matching on original graph. b_nodes will be unmatched
        :return:
            None
        """
        if blossom in mate:
            e = mate[blossom]
            cycle = set(blossom.cycle)
            if self.get_neighborhood(e.to_node) is blossom:
                in_node = e.to_node
                out_node = e.from_node
            else:
                in_node = e.from_node
                out_node = e.to_node

            while in_node not in cycle:
                in_node = self.b_map[in_node]

            while blossom.cycle[0] is not in_node:
                blossom.cycle.rotate(1)
                blossom.e_cycle.rotate(1)

            blossom.e_cycle = list(blossom.e_cycle)

            for node in blossom.cycle:
                self.b_map.pop(node)
                node.wake()

            for k in range(1, len(blossom.e_cycle), 2):
                mate[self.get_neighborhood(blossom.e_cycle[k].to_node)] = blossom.e_cycle[k]
                mate[self.get_neighborhood(blossom.e_cycle[k].from_node)] = blossom.e_cycle[k]
            mate[in_node] = mate[blossom]
            mate.pop(blossom)

        else:
            for node in blossom.cycle:
                self.b_map.pop(node)
                node.wake()

        blossom.kill()
        self.nodes.remove(blossom)
        return blossom


    def get_neighborhood(self, node):
        while node in self.b_map:
            node = self.b_map[node]
        return node

    def get_edge_val(self, node1, node2):
        if node1 not in self.b_map or node2 not in self.b_map:
            return 0
        n1 = []
        n2 = []
        while node1 in self.b_map:
            n1.append(node1)
            node1 = self.b_map[node1]

        while node2 in self.b_map:
            n2.append(node2)
            node2 = self.b_map[node2]

        if n1[-1] != n2[-1]:
            return 0
        i = 1
        l1 = len(n1)
        l2 = len(n2)
        while n1[l1 - i] is n2[l2 - i]:
            i += 1
        return sum([n.val for n in n1[:l1 - i + 2]])


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
