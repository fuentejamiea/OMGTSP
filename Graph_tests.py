import unittest
from My_Graph import *
from Matching import *
import numpy as np
import pickle
import gurobipy
import time


def random_cardinality_matching(n, num, denom, pickle_path=""):
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
            n, num, denom = map(int, pickle_path.split("_")[:3])
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
    mate, _, _, _, _ = maximal_matching(g1)
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

def random_weighted_matching(n, rand_range, pickle_path=""):
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
            n, rand_range = map(int, pickle_path.split("_")[:2])
    else:
        mat = []
    g1 = Graph()
    model = gurobipy.Model()
    model.setParam("OutputFlag", False)
    g1.add_nodes(n)
    con_map = {i: set() for i in range(n)}


    for i in range(n):
        if not pickle_path:
            new_row = np.random.randint(1, rand_range, i)
            mat.append(new_row)
        for j in range(i):
            n1 = g1.get_node(i)
            n2 = g1.get_node(j)
            g1.add_edge(n1, n2, mat[i][j])
            new_var = model.addVar(obj=mat[i][j], vtype=gurobipy.GRB.BINARY, name="({},{})".format(i, j))
            model.update()
            con_map[i].add(new_var)
            con_map[j].add(new_var)

    for nd, const in con_map.items():
        model.addConstr(gurobipy.quicksum(const) == 1)
    model.update()
    model.optimize()
    opt_val = model.getObjective().getValue()



    mate = weighted_matching(g1)
    grb_match = [v.VarName.split(',') for v in model.getVars() if v.x != 0]
    grb_match = sorted([tuple(sorted([int(m[0][1:]), int(m[1][:-1])],reverse= True
                                     )) for m in grb_match], key=lambda x: x[0])
    print(grb_match)


    my_match = {g1.get_edge(n1, n2) for n1, n2 in mate.items()}
    my_val = sum([e.weight for e in my_match])
    my_match = sorted(clean_matching(mate, lambda x: x.is_alive()), key=lambda x: x[0])
    print(my_match)
    print(opt_val, my_val)
    opt_flag = opt_val == my_val

    if not opt_flag:
        print("{}_{}_random matching failure".format(n,rand_range))
        if not pickle_path:
            with open('{}_{}_problem_mat.pkl'.format(n,rand_range), 'wb') as f:
                pickle.dump(mat, f)

    return opt_flag


class GraphMethods(unittest.TestCase):

    def test_nodes(self):
        jamie = Node(1)
        self.assertEqual(str(jamie), "N(1)")

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


class MatchingMethods(unittest.TestCase):

    def test_aps(self):
        graph = write_graph("Tests/Matching_test1.txt")
        mate, _, _, _, _ = maximal_matching(graph)
        matching = clean_matching(mate, lambda node: node.is_alive())
        self.assertEqual(len(matching), 6)

    def test_random_maximal(self):
        for n in [10, 15, 20, 50, 75, 100]:
            for num in [1, 3, 7]:
                self.assertEqual((0, 0), random_cardinality_matching(n, num, 10))

    def test_weighted_matching(self):
        graph = My_Graph.write_graph("Tests/weighted_matching.txt")
        mate = weighted_matching(graph)
        my_match = {graph.get_edge(n1, n2) for n1, n2 in mate.items()}
        my_val = sum([e.weight for e in my_match])
        self.assertEqual(my_val, 44)


    def test_random_weights(self):
        """
        for n in [10,20,30,50,70,100]:
            self.assertTrue(random_weighted_matching(n, 30))
            """

        self.assertTrue(random_weighted_matching(0, 0, "100_25_problem_mat.pkl"))




if __name__ == '__main__':
    unittest.main()
