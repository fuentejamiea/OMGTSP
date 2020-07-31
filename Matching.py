from My_Graph import *
from collections import deque


def blossom_set_partition(update):
    """
    :param update:
        set containing Nodes and Blossoms
    :return:
        set partitioned into set of nodes and set of blossoms, recurses to add sub_nodes of blossom
    """
    blossom_set = set()
    node_set = set()
    q = deque(update)
    while q:
        node = q.pop()
        if node.is_blossom():
            q.extend(node.cycle)
            if node.is_alive():
                blossom_set.add(node)
        else:
            node_set.add(node)
    return node_set, blossom_set


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


class Matching(Graph):
    def __init__(self, pathname=""):
        super(Matching, self).__init__(pathname)
        self.b_map = {}

    def add_blossom(self, cycle, e_cycle, inner):
        """
        :param cycle:
            list of Nodes in blossom in cycle
        :param e_cycle:
            list of cycle edges in order
        :return:
            Blossom added to graph
        """
        n = len(self.nodes)
        new_blossom = Blossom(n, cycle, e_cycle, len(cycle))
        self.nodes.append(new_blossom)

        for node1 in cycle:
            inner.pop(node1,None)
            self.b_map[node1] = new_blossom
            node1.kill()

        for node1 in cycle:
            if node1.is_blossom():
                new_blossom.members += node1.members - 1
            for edge in node1.edges:
                f_node = self.get_neighborhood(edge.from_node)
                t_node = self.get_neighborhood(edge.to_node)
                node2 = f_node if self.get_neighborhood(node1) is t_node else t_node
                if node2 is not new_blossom:
                    # add edge to blossom only if it leads out of blossom
                    new_blossom.edges.append(edge)

        return new_blossom

    def flower(self, mate):
        blossom = self.nodes[-1]
        while blossom.is_blossom():
            self.expand(blossom, mate)
            blossom = self.nodes[-1]

    def expand(self, blossom, mate):
        """
        :param blossom:
            blossom object to be removed from graph
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
            else:
                in_node = e.from_node

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
        """
        :param node:
            Node of interest
        :return:
            returns outermost blossom node is a member of. Node itself if it is alive
        """
        while not node.is_alive():
            node = self.b_map[node]
        return node

    def backtrack(self, edge, inner, outer, mate):
        """
        :param edge:
             Edge that connects two alternating path search trees
        :param inner:
             {Node : match Edge} for nodes on odd length alternating path from root node
        :param outer:
            {Node : non-match Edge} for nodes on even length alternating path from root node
        :param mate:
            {Node : mate Edge} dict. mutated by either blossom or alternating path
        :return:
            b_node if path search results in blossom shrinking else False
        """
        n1 = []
        e1 = [edge]
        n2 = []
        e2 = []
        node = self.get_neighborhood(edge.from_node)
        if node in inner:
            # shift one node over to start with outer
            n1.append(node)
            e1.append(inner[node])
            f_node = self.get_neighborhood(inner[node].from_node)
            t_node = self.get_neighborhood(inner[node].to_node)
            node = f_node if node is t_node else t_node

        while outer[node]:
            # back track to 1st root node
            n1.append(node)
            e1.append(outer[node])
            f_node = self.get_neighborhood(outer[node].from_node)
            t_node = self.get_neighborhood(outer[node].to_node)
            node = f_node if node is t_node else t_node
            n1.append(node)
            e1.append(inner[node])
            f_node = self.get_neighborhood(inner[node].from_node)
            t_node = self.get_neighborhood(inner[node].to_node)
            node = f_node if node is t_node else t_node
        # add stem
        n1.append(node)

        node = self.get_neighborhood(edge.to_node)
        if node in inner:
            n2.append(node)
            e2.append(inner[node])
            f_node = self.get_neighborhood(inner[node].from_node)
            t_node = self.get_neighborhood(inner[node].to_node)
            node = f_node if node is t_node else t_node

        while outer[node]:
            # back track to 2nd root node
            n2.append(node)
            e2.append(outer[node])
            f_node = self.get_neighborhood(outer[node].from_node)
            t_node = self.get_neighborhood(outer[node].to_node)
            node = f_node if node is t_node else t_node
            n2.append(node)
            e2.append(inner[node])
            f_node = self.get_neighborhood(inner[node].from_node)
            t_node = self.get_neighborhood(inner[node].to_node)
            node = f_node if node is t_node else t_node
        # add stem
        n2.append(node)

        if n1[-1] is n2[-1]:
            #blossom detected
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
            b_node = self.add_blossom(cycle, e_cycle, inner)
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
        return False

    def aps(self, mate, roots):
        """
        :param mate:
            {Node: mate Edge} mate Edges in current matching
            *mutated* augmented when augmenting path/blossom is found
        :param roots:
            unmatched Nodes that starts alternating path search
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
                        # apply blossom map
                        f_node = self.get_neighborhood(edge.from_node)
                        t_node = self.get_neighborhood(edge.to_node)
                        node2 = f_node if node1 is t_node else t_node
                        if node2 in outer:
                            # found augmenting path
                            blossom = self.backtrack(edge, inner, outer, mate)
                            if not blossom:
                                return False
                            q.append(blossom)
                            break

                        elif node2 not in inner:
                            inner[node2] = edge
                            q.append(node2)

            if node1 in inner:
                edge = mate[node1]
                f_node = self.get_neighborhood(edge.from_node)
                t_node = self.get_neighborhood(edge.to_node)
                node2 = f_node if node1 is t_node else t_node
                if node2 in inner:
                    # found augmenting path
                    blossom = self.backtrack(edge, inner, outer, mate)
                    if not blossom:
                        return False
                    q.append(blossom)

                elif node2 not in outer:
                    outer[node2] = edge
                    q.append(node2)

        return inner, outer

    def maximal_matching(self, mate=None, expand=True):
        """
        :param mate:
            optional initial matching
            *mutated* updated to matching of higher cardinality if one exists
        :param expand:
             bool determining if blossoms will be expanded after each aps
        :return:
            mate: Maximal cardinality matching on Graph by Edmond's Blossom algorithm
        """
        if mate is None:
            mate = {}
        maximal = False
        while not maximal:
            roots = [node for node in self.nodes if node.is_alive() and node not in mate]
            maximal = self.aps(mate, roots)
            if expand:
                self.flower(mate)

        return mate, maximal[0], maximal[1]

    def get_edge_val(self, node1, node2):
        """
        :param node1:
            1st Node of Edge object of interest
        :param node2:
            2nd Node of Edge object of interest
        :return:
            sum of all blossom.val values from all blossoms which contain both node1 Node and node2 Node
        """
        if node1.is_alive() or node2.is_alive():
            return 0
        n1 = []
        n2 = []
        while not node1.is_alive():
            n1.append(node1)
            node1 = self.b_map[node1]

        while not node2.is_alive():
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

    def weighted_matching(self):
        """
        :return:
            set of Edges corresponding to min weight perfect matching. Found using Edmond's primal-dual algorithm
            weight of matching found using dual variables. Should match sum of Edge weights from matching
        """

        for node in self.get_nodes():
            min_edge = min(node.edges, key=lambda edge: edge.weight)
            node.val = .5 * min_edge.weight

        n = len(self.nodes)
        mate = {}

        while True:
            for edge in self.edges:
                n1 = edge.to_node
                n2 = edge.from_node
                s_val = self.get_edge_val(n1, n2)
                if n1.val + n2.val + s_val == edge.weight:
                    # determine if edge is active
                    edge.wake()
                else:
                    edge.kill()

            # update matching, find blossom, get inner/outer node sets
            mate, inner, outer = self.maximal_matching(mate=mate, expand=False)
            if len(mate) == n:
                break
            # separate Nodes from alive blossoms
            inner_n, inner_b = blossom_set_partition(inner)
            outer_n, outer_b = blossom_set_partition(outer)

            delta1 = delta2 = delta3 = float('inf')

            # dual variable updates
            for edge in self.edges:
                n1 = edge.to_node
                n2 = edge.from_node
                n1_out = n1 in outer_n
                n1_in = n1 in inner_n
                n2_out = n2 in outer_n
                n2_in = n2 in inner_n
                if n1_out and n2_out and self.get_neighborhood(n1) is not self.get_neighborhood(n2):
                    delta1 = min(delta1, (edge.weight - n1.val - n2.val)/2)
                elif (n1_out and not (n2_in or n2_out)) or (n2_out and not (n1_in or n1_out)):
                    delta2 = min(delta2, edge.weight - n1.val - n2.val)

            for node in inner_b:
                delta3 = min(delta3, -node.val/2)
            delta = min(delta1, delta2, delta3)

            for node in inner_n:
                node.val -= delta

            for node in outer_n:
                node.val += delta

            j = -1
            blossom = self.nodes[j]
            while blossom.is_blossom():
                if blossom in outer_b:
                    blossom.val -= 2 * delta
                elif blossom in inner_b:
                    blossom.val += 2 * delta

                if blossom.val >= 0 and blossom.is_alive():
                    # expand newly invalid blossoms
                    self.expand(blossom, mate)
                else:
                    j -= 1
                blossom = self.nodes[j]

        s1 = sum([nodes.val for nodes in self.nodes[:n]])
        s2 = sum([b.val * ((b.members - 1)/2) for b in self.nodes[n:]])
        self.flower(mate)
        return mate, s1 + s2
