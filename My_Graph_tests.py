import unittest
from My_Graph import *


class GraphMethods(unittest.TestCase):
    def test_nodes(self):
        jamie = Node(1)
        carina = Node(2)
        impostor = Node(1)
        diksha = Node(3)

        node_set = {jamie, carina}
        self.assertEqual(len(node_set), 2)
        node_set.add(jamie)
        self.assertEqual(len(node_set), 2)
        node_set.add(impostor)
        self.assertEqual(len(node_set), 3)
        self.assertEqual(str(jamie), "N(1)")
        self.assertEqual(str(impostor), str(jamie))
        self.assertNotEqual(jamie, impostor)

        self.assertTrue(jamie < carina)
        self.assertTrue(diksha > jamie)
        self.assertTrue(carina < diksha)
        self.assertRaises(TypeError, lambda: carina <= jamie)
        self.assertRaises(TypeError, lambda: carina >= diksha)

    def test_edge(self):
        jamie = Node(1)
        carina = Node(2)
        friendship = Edge(jamie, carina, 100000)

        self.assertEqual(str(friendship), "(N(1)-(100000)-N(2))")

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


if __name__ == '__main__':
    unittest.main()
