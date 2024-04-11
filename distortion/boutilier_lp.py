from datetime import datetime

import pulp


class BoutilierOptimalDistortionLP(object):
    def __init__(self, pref):
        self.pref = pref
        self.lp_prob = None
        self.p_vars = None
        self.D = None
        self.n = len(self.pref)
        self.m = len(self.pref[0])

    def setup(self):
        # Create PuLP problem
        self.lp_prob = pulp.LpProblem("BoutilierDistortion", pulp.LpMaximize)

        # Define delta and P variables
        self.D = pulp.LpVariable('D', lowBound=0)
        self.p_vars = pulp.LpVariable.dicts("P", [c for c in range(self.m)], lowBound=0)
        y_vars = pulp.LpVariable.dicts(
            "y",
            [(i, c) for i in range(self.n) for c in range(self.m)])
        x_vars = pulp.LpVariable.dicts(
            "x",
            [(c_1, c_2) for c_1 in range(self.m) for c_2 in range(self.m)])
        z_vars = pulp.LpVariable.dicts(
            "z",
            [(k, j, c) for k in range(self.m - 1) for j in range(self.n) for c in range(self.m)], lowBound=0)

        # Add objective function to problem
        self.lp_prob += self.D

        # P sum up to 1
        self.lp_prob += pulp.lpSum([self.p_vars[i] for i in range(self.m)]) == 1

        # INEQ(p's, beta=D, pref, a^*)
        for selected_a in range(self.m):
            # sum y_{i,a^*} >= 0
            self.lp_prob += pulp.lpSum([y_vars[(i, selected_a)] for i in range(self.n)]) >= 0
            self.lp_prob += self.D + pulp.lpSum([x_vars[(a_2, selected_a)] for a_2 in range(self.m)]) <= 0
            for i in range(self.n):
                top_i = self.pref[i][0]
                self.lp_prob += self.p_vars[top_i] + x_vars[(top_i, selected_a)] - \
                                y_vars[(i, selected_a)] - z_vars[(0, i, selected_a)] >= 0
                for r in range(1, self.m - 1):
                    a = self.pref[i][r]
                    self.lp_prob += self.p_vars[a] + x_vars[(a, selected_a)] - \
                                    y_vars[(i, selected_a)] - z_vars[(r, i, selected_a)] + \
                                    z_vars[(r - 1, i, selected_a)] >= 0
                if self.m > 1:
                    bottom_i = self.pref[i][-1]
                    self.lp_prob += self.p_vars[bottom_i] + x_vars[(bottom_i, selected_a)] - \
                                    y_vars[(i, selected_a)] + z_vars[(self.m - 2, i, selected_a)] >= 0
            for a in range(self.m):
                if a != selected_a:
                    self.lp_prob += x_vars[(a, selected_a)] >= 0

        for c in range(self.m):
            self.lp_prob += self.p_vars[c] >= 0

    def solve(self, log_verbose=False):
        # Solve problem
        self.lp_prob.solve(pulp.GUROBI_CMD(options=[('LogToConsole', 0)]))

        # Return optimal values for P
        p_opt = [pulp.value(self.p_vars[c]) for c in range(self.m)]

        dist = 1 / pulp.value(self.D)

        if log_verbose:
            print(f"Boutilier et al. Result: D= {dist}, p= {p_opt}")

        return p_opt
