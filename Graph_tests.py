import unittest
from My_Graph import *
from make_instance import *


class GraphMethods(unittest.TestCase):

    def test_nodes(self):
        jamie = Node(1)
        self.assertEqual(str(jamie), "N(1)")

    def test_edge(self):
        jamie = Node(1)
        carina = Node(2)
        friendship = Edge(jamie, carina, 100000)

        self.assertEqual(str(friendship), "(1-(100000)-2)")

    def test_graph(self):
        friends = Graph()
        alexis, carina, jamie = friends.add_nodes(3)
        ship1 = friends.add_edge(alexis, carina, 10)
        ship3 = friends.add_edge(alexis, jamie, 30)
        ship2 = friends.add_edge(jamie, carina, 20)

        self.assertEqual(friends.nodes, [alexis, carina, jamie])
        self.assertTrue(isinstance(jamie, Node))
        self.assertIs(alexis, friends.get_node(0))
        self.assertEqual(friends.get_node(4), None)
        self.assertEqual(friends.edges, [ship1, ship3, ship2])
        friends.order_edges()
        self.assertEqual(friends.edges, [ship1, ship2, ship3])


class MatchingMethods(unittest.TestCase):

    def test_aps(self):
        graph = write_graph("TSP/Matching_test1.txt")
        mate, b_list = maximal_matching(graph)
        matching = clean_matching(mate, lambda node: node.is_alive())
        self.assertEqual(len(matching), 6)

    def test_blossom(self):
        graph = write_graph("TSP/blossom_test1.txt")
        mate = {}
        for edge in graph.edges:
            if edge.weight == 1:
                mate[edge.to_node] = edge.from_node
                mate[edge.from_node] = edge.to_node

        mate, b_list = maximal_matching(graph, mate)
        matching = clean_matching(mate, lambda node: node.is_alive())
        self.assertEqual(matching, {(18, 15), (14, 13), (12, 11), (17, 16)})

    def test_maximal_matching(self):
        for n in [10, 15, 25, 50, 100]:
            for num in [1, 3, 7]:
                self.assertEqual(random_matching_test(n, num, 10), (0, 0))


if __name__ == '__main__':
    unittest.main()
