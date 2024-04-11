import copy

import numpy as np


def sample_plackett_luce(_weights):
    weights = copy.deepcopy(_weights)
    m = len(weights)
    new_ranking = list()
    rem_ids = list(range(m))
    for i in range(m):
        normalized_weights = np.array(weights) / sum(weights)
        new_sample_ind = np.random.choice(len(rem_ids), p=normalized_weights)
        new_ranking.append(rem_ids[new_sample_ind])
        del rem_ids[new_sample_ind]
        del weights[new_sample_ind]
    return new_ranking


def generate_plackett_luce(n, m):
    weights = np.random.dirichlet(np.ones(m)).tolist()
    pref_profile = list()
    for i in range(n):
        new_pref = sample_plackett_luce(weights)
        pref_profile.append(new_pref)
    return pref_profile


if __name__ == '__main__':
    print(generate_plackett_luce(3, 2))

