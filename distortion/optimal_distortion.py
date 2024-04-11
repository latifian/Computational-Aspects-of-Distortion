import copy

import pulp


class NewOptimalDistortionLP(object):
    def __init__(self, pref, weights=None):
        self.pref = pref
        self.lp_prob = None
        self.p_vars = None
        self.delta_vars = None
        self.n = len(self.pref)
        self.m = len(self.pref[0])
        self.weights = copy.deepcopy(weights)

    def _add_variables(self):
        self.delta_vars = pulp.LpVariable.dicts(
            "delta",
            [(i, c) for i in range(self.n) for c in range(self.m)])
        self.p_vars = pulp.LpVariable.dicts("P", [c for c in range(self.m)], lowBound=0)

    def setup(self):
        # Create PuLP problem
        self.lp_prob = pulp.LpProblem("Distortion", pulp.LpMinimize)

        # Define delta and P variables
        self._add_variables()

        # Add objective function to problem
        self.lp_prob += pulp.lpSum([self.p_vars[c] for c in range(self.m)])

        # positive probabilities
        for c in range(self.m):
            self.lp_prob += self.p_vars[c] >= 0

        self._add_main_constraints()

        for c in range(self.m):
            if self.weights is None:
                self.lp_prob += pulp.lpSum([self.delta_vars[(i, c)] for i in range(self.n)]) <= 0
            else:
                self.lp_prob += pulp.lpSum([self.delta_vars[(i, c)] * self.weights[i] for i in range(self.n)]) <= 0

    def _add_main_constraints(self):
        # Add constraints to problem
        for i in range(self.n):
            for c in range(self.m):
                for k in range(1, self.m + 1):
                    self.lp_prob += self.delta_vars[(i, c)] >= int(self.pref[i].index(c) < k) / k - \
                                    pulp.lpSum([self.p_vars[self.pref[i][j]] for j in range(k)]) / k

    def solve(self, log_verbose=False):
        # Solve problem
        self.lp_prob.solve(pulp.GUROBI_CMD(options=[('LogToConsole', 0)]))

        # Return optimal values for P
        p_opt = [pulp.value(self.p_vars[c]) for c in range(self.m)]
        total_p = sum(p_opt)
        p_opt = [x / total_p for x in p_opt]

        if log_verbose:
            print(f"New LP Result: D= {total_p}, p= {p_opt}")

        return p_opt


class OptimizedNewOptimalDistortionLP(NewOptimalDistortionLP):
    def __init__(self, pref, weights=None):
        super().__init__(pref, weights)
        self.alpha_vars = None
        self.beta_vars = None
        self.ps_vars = None

    def _add_variables(self):
        super()._add_variables()
        self.alpha_vars = pulp.LpVariable.dicts(
            "alpha",
            [(i, c) for i in range(self.n) for c in range(self.m - 1)])
        self.beta_vars = pulp.LpVariable.dicts(
            "beta",
            [(i, c) for i in range(self.n) for c in range(self.m)])
        self.ps_vars = pulp.LpVariable.dicts(
            "ps",
            [(i, r) for i in range(self.n) for r in range(self.m)])

    def _add_main_constraints(self):
        for i in range(self.n):
            self.lp_prob += self.ps_vars[(i, 0)] == self.p_vars[self.pref[i][0]]
            for r in range(1, self.m):
                self.lp_prob += self.ps_vars[(i, r)] == self.ps_vars[(i, r - 1)] + self.p_vars[self.pref[i][r]]
            for r in range(self.m):
                if r == self.m - 1:
                    break
                if r == 0:
                    self.lp_prob += self.alpha_vars[(i, r)] == - 1 / (r + 1) * self.ps_vars[(i, r)]
                else:
                    self.lp_prob += self.alpha_vars[(i, r)] >= - 1 / (r + 1) * self.ps_vars[(i, r)]
                    self.lp_prob += self.alpha_vars[(i, r)] >= self.alpha_vars[(i, r - 1)]
            for r in range(self.m - 1, -1, -1):
                if r == self.m - 1:
                    self.lp_prob += self.beta_vars[(i, r)] == 1 / (r + 1) - 1 / (r + 1) * self.ps_vars[(i, r)]
                else:
                    self.lp_prob += self.beta_vars[(i, r)] >= 1 / (r + 1) - 1 / (r + 1) * self.ps_vars[(i, r)]
                    self.lp_prob += self.beta_vars[(i, r)] >= self.beta_vars[(i, r + 1)]
            for r in range(self.m):
                c = self.pref[i][r]
                if r > 0:
                    self.lp_prob += self.delta_vars[(i, c)] >= self.alpha_vars[(i, r - 1)]
                self.lp_prob += self.delta_vars[(i, c)] >= self.beta_vars[(i, r)]


def find_optimal_distribution(pref, weights=None):
    """

    :param pref: The preference profile of n voters over m candidates
        - Should be a list of n lists
        - Each inner list should be a permutation of [0, 1, ..., m - 1]
        where the i-th one denotes the preference ranking of the i-th voter
    :param weights: voters can have weights
        - should be a list of n numbers
        - 'None' is equivalent equal weights
    :return: the distribution that minimizes the worst-case distortion for the given instance
    """
    ins = OptimizedNewOptimalDistortionLP(pref, weights)
    ins.setup()
    return ins.solve()
