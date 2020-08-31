import unittest
from collections import Counter
from Heuristic import *
from My_Graph import *


class HeuristicMethods(unittest.TestCase):
    def test_MST(self):
        graph = Graph("tests/MST_test.txt")
        mst, weight = min_spanning_tree(graph)
        self.assertEqual(len(mst), len(graph.nodes) - 1)
        self.assertEqual(weight, 37)

    def test_euler(self):
        graph = Graph("tests/Euler_test.txt")
        tour = eulerian_tour(graph)
        count = Counter(tour)
        for node in graph.nodes:
            self.assertEqual(count[node], len(node.edges)/2)

    def test_christofides(self):
        graph = Graph("instances/ulysses22.txt")
        tour, weight = christofides(graph)
        print(tour)
        self.assertEqual(set(node.num for node in tour), set(range(len(graph.nodes))))

