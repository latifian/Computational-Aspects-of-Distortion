import copy
import os
import pickle
from tqdm import tqdm


def get_preflib_directory():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    relative_preflib_path = '../datasets/preflib-soc'
    return os.path.normpath(os.path.join(current_directory, relative_preflib_path))


PREFLIB_PATH = get_preflib_directory()


def get_pref(preflib_ds_path):
    profile = []
    with open(preflib_ds_path) as f:
        n, m = None, None
        unique_n = None
        while True:
            line = f.readline()
            if not line.startswith('#'):
                break
            if "NUMBER ALTERNATIVES" in line:
                m = int(line.split(' ')[-1])
            if "NUMBER VOTERS" in line:
                n = int(line.split(' ')[-1])
            if "NUMBER UNIQUE ORDERS" in line:
                unique_n = int(line.split(' ')[-1])

        assert None not in [n, m, unique_n]

        for i in range(unique_n):
            n_voters, pref_str = tuple(line.split(': '))
            n_voters = int(n_voters)
            pref = [int(x) - 1 for x in pref_str.split(',')]
            assert len(pref) == m
            profile.extend([copy.deepcopy(pref) for _ in range(int(n_voters))])
            line = f.readline()

    assert n == len(profile)
    return profile


def parse_all_preflib_datasets():
    all_files = list(os.listdir(PREFLIB_PATH))
    all_preflib_ds = list()
    for preflib_ds_file in tqdm(all_files):
        profile = get_pref(os.path.join(PREFLIB_PATH, preflib_ds_file))
        all_preflib_ds.append(profile)
    with open(os.path.join(PREFLIB_PATH, 'preflib.pkl'), 'wb') as f:
        pickle.dump(all_preflib_ds, f)


def load_dumped_preflib_datasets():
    with open(os.path.join(PREFLIB_PATH, 'preflib.pkl'), 'rb') as f:
        all_preflib_ds = pickle.load(f)
    return all_preflib_ds
