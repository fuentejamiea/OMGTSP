import unittest
from My_Graph import *
from Matching import *
import numpy as np
import pickle
import gurobipy


def random_matching_test(n, num, denom, pickle_path=""):
    """
    :param n:
        number of Nodes in test Graph
    :param num:
        numerator in num/denom = probability there exists an edge between any two distinct nodes
    :param denom:
        denominator in num/denom = probability there exists an edge between any two distinct nodes
    :param pickle_path:
        (optional) path to pickled random mat representing node adj. matrix
    :return:
        diff = difference of cardinality of maximal matching algorithm and cardinality of matching found by gurobi ILP
        invalid = number edges in matching from algo not in graph
    """
    if pickle_path:
        with open(pickle_path, 'rb') as f:
            mat = pickle.load(f)
            n = len(mat)
    else:
        mat = []
    cutoff = denom - num
    g1 = Graph()
    model = gurobipy.Model()
    model.setParam("OutputFlag", False)
    g1.add_nodes(n)
    con_map = {i: set() for i in range(n)}


    for i in range(n):
        if not pickle_path:
            new_row = np.random.randint(0, denom, i)
            mat.append(new_row)
        for j in range(i):
            if mat[i][j] >= cutoff:
                n1 = g1.get_node(i)
                n2 = g1.get_node(j)
                n1.val += 1
                n2.val += 1
                g1.add_edge(n1, n2, 0)
                new_var = model.addVar(obj=-1, vtype=gurobipy.GRB.BINARY, name="({},{})".format(i, j))
                model.update()
                con_map[i].add(new_var)
                con_map[j].add(new_var)

    for nd, const in con_map.items():
        model.addConstr(gurobipy.quicksum(const) <= 1)
    model.update()
    mate, _ = maximal_matching(g1)
    expand(g1, mate)
    matching = clean_matching(mate, lambda node: node is not None)
    count1 = len(matching)
    matching = {m for m in matching if mat[m[0]][m[1]] >= cutoff}
    count2 = len(matching)
    invalid = count1 - count2
    model.optimize()
    grb_match = [v.VarName for v in model.getVars() if v.x != 0]
    diff = len(matching) - len(grb_match)
    if (diff != 0 or invalid != 0) and not pickle_path:
        with open('{}_{}_{}_problem_mat.pkl'.format(n,num,denom), 'wb') as f:
            pickle.dump(mat, f)
    return diff, invalid


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


class MatchingMethods(unittest.TestCase):

    def test_aps(self):
        graph = write_graph("Tests/Matching_test1.txt")
        mate,outer = maximal_matching(graph)
        matching = clean_matching(mate, lambda node: node.is_alive())
        self.assertEqual(len(matching), 6)

    def test_blossom(self):
        graph = write_graph("Tests/blossom_test1.txt")
        mate = {}
        for edge in graph.edges:
            if edge.weight == 1:
                mate[edge.to_node] = edge.from_node
                mate[edge.from_node] = edge.to_node

        mate, outer = maximal_matching(graph, mate)
        matching = clean_matching(mate, lambda node: node.is_alive())
        self.assertEqual(matching, {(18, 15), (14, 13), (12, 11), (17, 16)})

    def test_maximal_matching(self):
        for n in [10, 15, 20, 50, 75, 100]:
            for num in [1, 3, 7]:
                self.assertEqual(random_matching_test(n, num, 10), (0, 0))

    def test_problem_mat(self):
        self.assertEqual(random_matching_test(10, 3, 10, "50_1_10_problem_mat.pkl"), (0, 0))




if __name__ == '__main__':
    unittest.main()