from datetime import datetime
import time
import copy
import pickle
import numpy as np
from numpy import testing
from tqdm import tqdm

from data_generation.mallows import sample_mallow
from data_generation.plackett_luce import generate_plackett_luce
from data_generation.preflib import load_dumped_preflib_datasets
from data_generation.single_peaked import generate_single_peaked
from distortion.boutilier_lp import BoutilierOptimalDistortionLP
from distortion.optimal_distortion import OptimizedNewOptimalDistortionLP

optimal_decision_rules = [
    {
        'class': OptimizedNewOptimalDistortionLP,
        'name': 'optimized-new-lp'
    }
    ,
    {
        'class': BoutilierOptimalDistortionLP,
        'name': 'boutilier'
    }
]


class RuntimeComparisonExperiment(object):
    def __init__(self, suffix_name=None):
        self.datasets = list()
        self.decision_rules = copy.deepcopy(optimal_decision_rules)
        self.experiment_results = dict()
        self.experiment_name = f'runtime_exp_{np.random.randint(1e9, 1e10)}'
        if suffix_name:
            self.experiment_name += suffix_name

    def add_datasets(self, ds):
        ds['id'] = len(self.datasets)
        self.datasets.append(ds)

    def add_decision_rule(self, rule):
        self.decision_rules.append(rule)

    def _run(self, rule, ds):
        ds_id, rule_name = ds['id'], rule['name']
        res_key = ds_id, rule_name
        if res_key in self.experiment_results:
            print(f"already computed: ds_id: {ds_id}, rule_name: {rule_name}")
            # already computed
            return False

        r_solver = rule['class'](ds['pref'])
        st_time_setup = time.time()
        print("Here before setup", datetime.now())
        r_solver.setup()
        print("Here after setup", datetime.now())
        st_time_solve = time.time()
        p = r_solver.solve(log_verbose=False)
        en_time_solve = time.time()
        print("SOLVED", datetime.now())
        exp_res = {
            'solve_time': en_time_solve - st_time_solve,
            'setup_time': st_time_solve - st_time_setup,
            'total_time': en_time_solve - st_time_setup,
            'distribution': p,
            'rule': rule_name,
            'ds_id': ds_id,
            'n': len(ds['pref']),
            'm': len(ds['pref'][0])
        }
        print(f"Completed experiment: {res_key}: solve_time: {exp_res['solve_time']}"
              f", total_time: {exp_res['total_time']}")  # \n \t P={p}")
        self.experiment_results[res_key] = exp_res
        return True

    def _save(self):
        with open(f'{self.experiment_name}_results.pkl', 'wb') as f:
            pickle.dump(self.experiment_results, f)

    def load_results(self):
        with open(f'{self.experiment_name}_results.pkl', 'rb') as f:
            self.experiment_results = pickle.load(f)

    def save_datasets(self):
        with open(f'{self.experiment_name}_datasets.pkl', 'wb') as f:
            pickle.dump(self.datasets, f)

    def load_datasets(self):
        with open(f'{self.experiment_name}_datasets.pkl', 'rb') as f:
            self.datasets = pickle.load(f)

    def run_rules_on_datasets(self):
        any_change = False
        for ds in self.datasets:
            _n, _m = len(ds['pref']), len(ds['pref'][0])
            ds_id = ds['id']
            print(f"\n\n@DS {ds_id} with n: {_n} and m: {_m}\n\n")
            for r_id, rule in enumerate(self.decision_rules):
                any_change |= self._run(rule, ds)
                if r_id > 0:
                    p0 = self.experiment_results[(ds_id, self.decision_rules[0]['name'])]['distribution']
                    p1 = self.experiment_results[(ds_id, rule['name'])]['distribution']
                    for i in range(_m):
                        testing.assert_almost_equal(p0[i], p1[i], decimal=5)
            if any_change:
                self._save()
                any_change = False
        return any_change


def run_mallows_experiments():
    n = 1000
    exp_instance = RuntimeComparisonExperiment(suffix_name='m_10_60_each_10_mallows')
    for m in range(10, 61, 10):
        for phi in [0.5, 0.1, 0.2, 0.8, 1]:
            for i in range(10):
                print("AT:", n, m, phi, i)
                new_pref = sample_mallow(n, m, phi)
                new_ds = {
                    'pref': new_pref,
                    'model': 'mallows',
                    'phi': phi
                }
                exp_instance.add_datasets(new_ds)
    exp_instance.save_datasets()
    exp_instance.run_rules_on_datasets()


def run_plackett_luce_experiments():
    n = 1000
    exp_instance = RuntimeComparisonExperiment(suffix_name='_m_10_60_each_10_plackett_luce')
    for m in range(10, 61, 10):
        for i in range(10):
            print("AT:", n, m, i)
            new_pref = generate_plackett_luce(n, m)
            new_ds = {
                'pref': new_pref,
                'model': 'plackett_luce',
            }
            exp_instance.add_datasets(new_ds)
    exp_instance.save_datasets()
    exp_instance.run_rules_on_datasets()


def run_single_peaked():
    n = 1000
    exp_instance = RuntimeComparisonExperiment(suffix_name='_m_10_60_each_10_single_peaked')
    for m in range(10, 61, 10):
        for i in range(10):
            print("AT:", n, m, i)
            new_pref = generate_single_peaked(n, m)
            new_ds = {
                'pref': new_pref,
                'model': 'single_peaked',
            }
            exp_instance.add_datasets(new_ds)
    exp_instance.save_datasets()
    exp_instance.run_rules_on_datasets()


def run_preflib_experiments():
    exp_instance = RuntimeComparisonExperiment(suffix_name='preflib')
    preflib_ds = load_dumped_preflib_datasets()
    preflib_ds = sorted(preflib_ds, key=lambda x: len(x) * (len(x[0]) ** 2))
    for ds in tqdm(preflib_ds):
        new_ds = {
            'pref': ds,
            'model': 'preflib',
        }
        exp_instance.add_datasets(new_ds)
    exp_instance.save_datasets()
    exp_instance.run_rules_on_datasets()


def continue_experiment(exp_name):
    r_exp = RuntimeComparisonExperiment()
    r_exp.experiment_name = exp_name
    r_exp.load_datasets()
    r_exp.load_results()
    r_exp.run_rules_on_datasets()


if __name__ == '__main__':
    run_mallows_experiments()
