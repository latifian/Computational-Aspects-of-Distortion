
def check_distortion_is_at_most(d, pref, p):
    m = len(pref[0])

    dist_per_candidate = [0 for _ in range(m)]

    for i, pref_i in enumerate(pref):
        partial_sum_p = [None for _ in range(m)]
        for ind, c in enumerate(pref_i):
            partial_sum_p[ind] = d * p[c]
            if ind > 0:
                partial_sum_p[ind] += partial_sum_p[ind - 1]

        ps_partial_max = [None for _ in range(m)]
        for ind in range(m):
            ps_partial_max[ind] = -partial_sum_p[ind] / (ind + 1)
            if ind > 0:
                ps_partial_max[ind] = max(ps_partial_max[ind], ps_partial_max[ind - 1])

        reverse_partial_max = -d * 100
        for rev_ind, c in enumerate(reversed(pref_i)):
            ind = m - rev_ind - 1
            current_delta_i = (1 - partial_sum_p[ind]) / (ind + 1)
            reverse_partial_max = max(current_delta_i, reverse_partial_max)
            dist_per_candidate[c] += max(ps_partial_max[ind], reverse_partial_max)

    return max(dist_per_candidate) <= 0


def find_distortion_of_distribution(pref, p):
    m = len(pref[0])

    if not check_distortion_is_at_most(m * m, pref, p):
        return m * m

    lo_d, hi_d = 0.0, 1.0 * m * m

    for binary_search_iter in range(100):
        mid_d = (lo_d + hi_d) / 2
        if check_distortion_is_at_most(mid_d, pref, p):
            hi_d = mid_d
        else:
            lo_d = mid_d
    return hi_d
