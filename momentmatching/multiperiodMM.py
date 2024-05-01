import pandas as pd
import numpy as np
import os
import json
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.config import Config
from checkarbitrage import check_arbitrage
from oneperiodMM import MomentMatching, solveGA

file_name = os.path.join(
        '.', 'cfg', 'scenariogeneration_settings.json'
    )
fp = open(file_name, 'r')
settings = json.load(fp)
fp.close()

time_horizon = settings['time_horizon']
num_azioni = settings['num_azioni']
branching_factors = settings['branching_factors']
exp_mean = settings['exp_mean']
exp_std = settings['exp_std']
exp_skew = settings['exp_skew']
exp_kur = settings['exp_kur']
exp_cor = settings['exp_cor']
volclumping = settings['volclumping']
rf_rate = settings['rf_rate']
risk_premium = settings['risk_premium']

size_pop = 200
scenario_tree = {}

new_exp_mean = np.empty(num_azioni)
new_exp_std = np.empty(num_azioni)
to_gen = branching_factors[0]

for t in range(time_horizon):
    p = np.full(branching_factors[t], 1/branching_factors[t])  
    if t == 0:
        problem = MomentMatching(num_azioni, branching_factors[t], p, exp_mean, exp_std, exp_skew, exp_kur, exp_cor)
        sol = solveGA(problem, size_pop)
        scenario_tree[t] = sol
        last_time_nodes = branching_factors[0]
    else:
        old_scenarios = branching_factors[t-1]
        to_gen = branching_factors[t] * last_time_nodes
        generated_scenarios = np.empty((num_azioni, to_gen))  
        for i in range(old_scenarios):
            for j in range(num_azioni):
                old_values = scenario_tree[t-1]
                old_value = old_values[j,i]
                new_exp_std[j] = volclumping[j]*(old_value-exp_mean[j]) + (1-volclumping[j])*exp_std[j]
                new_exp_mean[j] = rf_rate + risk_premium[j]*new_exp_std[j]

            problem = MomentMatching(num_azioni, branching_factors[t], p, new_exp_mean, new_exp_std, exp_skew, exp_kur, exp_cor)
            sol = solveGA(problem, size_pop)
            for s in range(branching_factors[t]):
                generated_scenarios[:, s+i] = sol