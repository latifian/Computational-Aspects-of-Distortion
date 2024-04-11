import numpy as np


def generate_single_peaked(n, m):
    voter_pos = np.random.rand(n)
    alts_pos = np.random.rand(m)

    rankings = []
    for i in range(n):
        absolute_diff = np.abs(alts_pos - voter_pos[i])
        ranking = np.argsort(absolute_diff)
        rankings.append(list(ranking))

    return rankings


if __name__ == '__main__':
    print(generate_single_peaked(3, 5))
