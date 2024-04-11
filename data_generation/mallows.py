# Credit to "https://github.com/ekhiru/top-k-mallows/blob/master/mallows_kendall.py"


import numpy as np


def v_to_ranking(v, n):
    """This function computes the corresponding permutation given a decomposition vector.
        Parameters
        ----------
        v: ndarray
            Decomposition vector, same length as the permutation, last item must be 0
        n: int
            Length of the permutation
        Returns
        -------
        ndarray
            The permutation corresponding to the decomposition vectors.
    """
    rem = list(range(n))
    rank = np.full(n, np.nan)
    for i in range(len(v)):
        rank[i] = rem[v[i]]
        rem.pop(v[i])
    return rank.astype(int)


# m rankings of size n
def sample_mallow(m, n, phi=0.5):
    if phi == 1.0:
        rankings = []
        for i in range(m):
            rankings.append(list(np.random.permutation(n)))
        return rankings

    theta = -np.log(phi)

    theta = np.full(n - 1, theta)

    s0 = np.array(range(n))

    rnge = np.array(range(n - 1))

    psi = (1 - np.exp((-n + rnge) * (theta[rnge]))) / (1 - np.exp(-theta[rnge]))
    vprobs = np.zeros((n, n))
    for j in range(n - 1):
        vprobs[j][0] = 1.0 / psi[j]
        for r in range(1, n - j):
            vprobs[j][r] = np.exp(-theta[j] * r) / psi[j]
    sample = []
    for samp in range(m):
        v = [np.random.choice(n, p=vprobs[i, :]) for i in range(n - 1)]
        v += [0]
        ranking = v_to_ranking(v, n)
        sample.append(ranking)

    sample = [list(s[s0]) for s in sample]

    return sample
