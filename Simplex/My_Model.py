import numpy as np


class MyModel:
    def __init__(self, m, n, obj=None, rhs=None, A=None):
        # TODO: change to sparse matrices after debugging complete
        self.m = m
        self.n = n
        if obj is None:
            self.obj = np.zeros((n,))
        else:
            self.obj = obj
        if rhs is None:
            self.rhs = np.zeros((m,))
        else:
            self.rhs = rhs
        if A is None:
            self.A = np.zeros((m, n))
        else:
            self.A = A
        self.x = np.zeros((n,))
        self.basic = []
        # TODO: add bounds when standard form simplex complete

    def compute_bfs(self):
        B = np.eye(self.m)
        c_n = np.zeros(self.n)
        c_b = np.ones(self.m)
        self.basic = [False] * self.n + [True] * self.m
        x_b = self.rhs
        d_n = c_n - c_b@self.A
        e = np.zeros((self.m,))
        entering = 0
        iter = 0
        while iter < 25 and d_n[entering] < 0:
            y = np.linalg.solve(B, self.A[:, entering])
            leaving = min(range(self.m), key=lambda i: x_b[i]/y[i] if y[i] > 0 else float('inf'))
            theta = x_b[leaving]/y[leaving]
            x_b = x_b - theta * y
            x_b[leaving] = theta

            e[leaving] = 1
            z = np.linalg.solve(B.T, e)
            e[leaving] = 0

            x = z @ (self.A)
            ratio = d_n[entering] / x[entering]
            d_n = d_n - (x * ratio)
            d_n[entering] = -ratio * y @ B[:, leaving]
            print(iter, entering, leaving)
            hold = B[:, leaving]
            B[:, leaving] = self.A[:, entering]
            self.A[:, entering] = hold
            entering = np.argmin(d_n)
            iter += 1

        print(B@x_b)







    def solve(self):
        non_basic = list(filter(lambda i: self.x[i] != 0, range(len(self.x))))
        B = self.A[:, non_basic]
        print(B.shape)
        print(np.linalg.det(B))










