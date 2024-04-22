
import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from itertools import chain, combinations
import time
import copy
import nashpy as nash
import sys
# print (sys.version)

def powerset(iterable):
    s = list(iterable)
    ret = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(list(x) for x in ret)

def renormalize(scores):
    norm = sum(scores)
    ret = [x/norm for x in scores]
    ret.sort(reverse=True)
    return ret

def solve_quadratic_program(U, P, log=False):
    n, m = len(U), len(U[0])

    # Create a new model
    model = gp.Model("quadratic_program")
    # model.params.LogToConsole = 0
    model.params.NonConvex = 2
    # model.params.MIPGap = 0.01
    print(model.getParamInfo("MIPGap"), flush=True)
    # Create variables q[i]
    q = model.addVars(n, lb=0, name="q")
    Z = model.addVar(lb=0, name="z")

    # Define SW(i) expressions
    SW = [gp.quicksum(U[j][i] * q[j] for j in range(n)) for i in range(m)]
    Prob = [gp.quicksum(P[j][i] * q[j] for j in range(n)) for i in range(m)]

    C = np.matmul(np.array(U), np.array(P).transpose())


    # Add linear constraints
    for i in range(m):
        model.addConstr(SW[i] >= 0)
        # if i > 0:
        #     model.addConstr(SW[i] <= SW[i-1])
        model.addConstr(SW[i] <= SW[0])
    # model.addConstr(SW[0] >= 1)


    for i in range(2, m):
        model.addConstr(Prob[i] <= Prob[i-1])


    model.addConstr(gp.quicksum(C[i][j]*q[i]*q[j] for i in range(n) for j in range(n)) <= Z*SW[0])
    model.addConstr(gp.quicksum(q[i] for i in range(n)) == 1)
    # model.addConstr(gp.quicksum(q[i] for i in range(n)) >= 1)
    model.update()
    # Define the quadratic objective expression
    # obj_expr = gp.quicksum(C[i][j]*q[i]*q[j]*Z[1] for i in range(n) for j in range(n))


    # q_np = np.array([q[i] for i in range(n)])

    model.addConstr(Z <= 1/3.42)
    model.addConstr(Z >= 1/10)
    model.setObjective(Z, GRB.MINIMIZE)

    # print(obj_expr, expr)
    # Optimize the model
    model.optimize()

    # Get the optimal solution
    optimal_q = [q[i].x for i in range(n)]
    norm = sum(optimal_q)
    print(model.objVal, Z.x)
    optimal_obj_value = norm/(model.objVal)
    # print("norm: {}, obj: {}, n/o: {}".format(norm, model.objVal, optimal_obj_value))
    sw = [float(SW[i].getValue()) for i in range(m)]
    sw_rank = "-".join([str(x) for x in sorted(range(len(sw)), key=lambda k: sw[k])])
    pr = [float(Prob[i].getValue()) for i in range(m)]
    pr_rank = "-".join([str(x) for x in sorted(range(len(pr)), key=lambda k: pr[k])])



    # print("{:.3f} =  {:.2f} * {:.2f}  +  {:.2f} * {:.2f}  +  {:.2f} * {:.2f}".format(model.objVal/norm, sw[0], pr[0], sw[1], pr[1], sw[2], pr[2]))
    # if model.objVal/norm < 1/3:
    #     print("Bad")


    if log:
        print( " "*11 +"  ".join(["Sw[{}]={:.2f}".format(i, sw[i]) for i in range(m)]))

        print(" "*11 +"  ".join(["Pr[{}]={:.2f}".format(i, pr[i]) for i in range(m)]))

        print(" "*10 +"-"*(12*m+10))

        X = [sw[i]*pr[i] for i in range(m)]
        print("  E[sw] = "+"+".join(["   {:.3f}   ".format(X[i]) for i in range(m)]) + "= {:.6f}".format(sum(X)))
        print( " "*(12*m-1) + "Sw[opt]  "+" = {:.6f}".format(max(sw)))

        print(" "*(12*m-1) + "   Dist  "+ " = {:.6f}     1/Pr[0]= {:.6f}".format(optimal_obj_value, 1/pr[0]))

        print(" "*(12*m+1) +"-"*(19))


        print("opt: {}".format(model.objVal))

    # print("Social welfares: {}".format(["{:.2f}".format(Prob[i].getValue()/norm for i in range(m))]))
    # print(sum(optimal_q))
    return [optimal_q[i] for i in range(n)], optimal_obj_value, sw_rank, pr_rank

def add_opt(ranking, k):
    first = ranking.copy()
    first.insert(k, 0)
    if k == 0:
        return [(first, k+1)]
    second = ranking.copy()
    m = len(ranking)

    second.append(0)
    if k == m:
        return [(first, k+1)]
    return [(first, k+1), (second, k)]

def generate_all(m):
    cands = [i for i in range(1, m)]
    # print(cands)
    all_rankings = []
    for rank in itertools.permutations(cands):
        for i in range(0, m):
            all_rankings.extend(add_opt(list(rank), i))
    return all_rankings

def short_generate_all(m):
    cands = [i for i in range(1, m)]
    # print(cands)
    all_rankings = []
    pws = powerset(cands)


    for s in pws:
        rank = []
        for c in cands:
            if c in s:
                rank.append(c)
        for c in cands:
            if c not in s:
                rank.append(c)
        temp = add_opt(rank, len(s))
        # print(s, temp)
        all_rankings.extend(temp)
    return all_rankings


def generate_U_and_P(all, p):
    m = len(all[0][0])
    U = []
    P = []
    for x in all:
        temp_u = [0]*m
        temp_p = [0]*m
        rank = x[0]
        k = x[1]

        for i in range(m):
            if i < k:
                temp_u[rank[i]] = 1/k
            temp_p[rank[i]] = p[i]
        U.append(temp_u)
        P.append(temp_p)
    return U, P

def best_rule_on_set_of_instances(m, instances):
    model = gp.Model("best_rule")
    model.params.LogToConsole = 0
    q = model.addVars(m, lb=0, name="q")

    one_over_D = model.addVar(lb=0, name="z")
    model.addConstr(gp.quicksum([q[i] for i in range(m)]) == 1)

    for i in range(1, m):
        model.addConstr(q[i-1] >=  q[i])
    for instance in instances:
        P = [0]*m
        SW = [0]*m
        for voter_type in instance:
            z = voter_type[1]
            pref, k = voter_type[0]
            for i in range(m):
                P[pref[i]] += q[i]*z
                # print("P[{}] += {}*{} ---> {} += {} = {}".format(pref[i], vec[i], q, P[pref[i]] - vec[i]*q, vec[i]*q, P[pref[i]]))
                if i < k:
                    SW[pref[i]] += 1/k*z

        Esw = sum([P[i]*SW[i] for i in range(m)])

        model.addConstr(one_over_D <= Esw/SW[0])
    model.setObjective(one_over_D, GRB.MAXIMIZE)

    model.optimize()

    # Get the optimal solution
    optimal_q = [q[i].x for i in range(m)]
    return 1/model.objVal, optimal_q

def find_worst_instance(probs, mode="short", log=False):

    if mode == "short":
        all_rankings = short_generate_all(len(probs))
    else:
        all_rankings = generate_all(len(probs))

    U, P = generate_U_and_P(all_rankings, probs)

    q1, dist1, sw_rank, pr_rank = solve_quadratic_program(U, P, log)

    if log:
        for i in range(len(all_rankings)):
            print("{}: {:.4f}".format(all_rankings[i], q1[i]))

        print("\nDistortion: {:.6f}".format(dist1))
    # print("dist: ", dist1)
    return dist1, [[all_rankings[i], q1[i]] for i in range(len(q1))]


def find_dist(probs, log=False):
    all_rankings = short_generate_all(len(probs))

    U, P = generate_U_and_P(all_rankings, probs)

    q1, dist1, sw_rank, pr_rank = solve_quadratic_program(U, P, log)

    return dist1

def dist_of_rule_on_input(vec, q_and_rankings):
    m = len(vec)
    P = [0]*m
    SW = [0]*m
    for voter_type in q_and_rankings:
        q = voter_type[1]
        pref, k = voter_type[0]
        for i in range(m):
            P[pref[i]] += vec[i]*q
            # print("P[{}] += {}*{} ---> {} += {} = {}".format(pref[i], vec[i], q, P[pref[i]] - vec[i]*q, vec[i]*q, P[pref[i]]))
            if i < k:
                SW[pref[i]] += 1/k*q
    # print(P)
    # print(SW)

    Esw = sum([P[i]*SW[i] for i in range(m)])

    # print()
    return SW[0]/Esw

def solve_game(matrix):
    # print("Matrix")
    # for x in matrix:
    #     print(x)
    m = len(matrix[0])
    game = nash.Game(np.array(matrix))
    out = game.linear_program()
    rule = [0]*m
    # print(out)
    for i in range(m):
        q = out[1][i]
        for k in range(i+1):
            rule[k] += q/(i+1)

    # print(rule, -1/game[out][0])
    return -1/game[out][0], rule

def matrix_game_iteration(m, precision, starting_set = []):
    if len(starting_set) == 0:
        working_set = [find_worst_instance([0.7, 0.2, 0.1]+[0]*(m-3))[1]]
    else:
        working_set = copy.deepcopy(starting_set)
    delta = 10
    all_rules = [[1/i]*i+[0]*(m-i) for i in range(1, m+1)]
    # print(all_rules)
    matrix = []
    for instance in working_set:
        temp = []
        for rule in all_rules:
            # print(rule, dist_of_rule_on_input(rule, instance))
            temp.append(-1/dist_of_rule_on_input(rule, instance))
        matrix.append(temp)
    previous_dist, opt_rule = solve_game(matrix)
    while delta > precision:
        dist_rule, new_instance = find_worst_instance(opt_rule)
        # print("New: ", new_instance)
        working_set.append(new_instance)
        temp = []
        for rule in all_rules:
            temp.append(-1/dist_of_rule_on_input(rule, new_instance))
        matrix.append(temp)
        new_dist, opt_rule = solve_game(matrix)
        print("------ Dist LB: {:.7f}, New Rule: {}, Dist previous rule: {}".format(new_dist, opt_rule, dist_rule), flush=True)
        print("Current time: ", time.time()-start, flush=True)
        sys.stdout.flush()
        # if time.time()-start > 4*3600:
        #     print(working_set)
        delta = abs(previous_dist - new_dist)
        # print(delta, "\n")
        previous_dist = new_dist
    return opt_rule, previous_dist


def lp_iteration(m, precision, starting_set = []):
    delta = 10
    if len(starting_set) == 0:
        starting_set = [find_worst_instance([1]+[0]*(m-1))[1]]
    working_set = copy.deepcopy(starting_set)
    previous_dist, opt_rule = best_rule_on_set_of_instances(m, working_set)
    # print(opt_rule)
    while delta > precision:
        dist_rule, new_instance = find_worst_instance(opt_rule)
        working_set.append(new_instance)
        new_dist, opt_rule = best_rule_on_set_of_instances(m, working_set)
        print("------ Dist LB: {:.7f}, New Rule: {}, Dist previous rule: {}".format(new_dist, opt_rule, dist_rule))
        print("Current time: ", time.time()-start)
        if time.time()-start > 2*3600:
            print(working_set)
        delta = abs(previous_dist - new_dist)
        # print(delta, "\n")
        previous_dist = new_dist
    return opt_rule, previous_dist, working_set

h = [0, 0] + [sum([1/i for i in range(1, m+1)]) for m in range(2, 11)]
all_vec = [0]*2 + [{"Plurality": [1]+[0]*(m-1),
           "Borda": [m-i-1 for i in range(m)],
        #    "2-App": [1]*2 + [0]*(m-2),
        #    "3-App": [1]*3 + [0]*(m-3),
           "Veto": [1]*(m-1) + [0],
           "Harmonic": [1/i for i in range(1, m+1)],
           "Golden Rule": [1/i + h[m]/m for i in range(1, m+1)]} for m in range(2, 11)]

m = 10
start = time.time()
rule_10 = renormalize([0.6275836446505112, 0.23618265481966041, 0.1362337005298284, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# print(find_worst_instance(rule_10))
print(time.time()-start)
# sleep(1)


print(find_worst_instance(rule_10), flush=True)
print(time.time(), flush=True)

starting_set_10.append(instance)
starting_set_10.append(instance2)

start = time.time()
rule, dist_mx = matrix_game_iteration(m, 0.01, starting_set_10)
print("Rule: {}, LB: {}, Dist: {}".format(rule, dist_mx, find_dist(rule)))
print("Matrix time: ", time.time()-start)

