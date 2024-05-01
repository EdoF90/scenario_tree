import numpy as np
import logging
import os
from oneperiodMM import MomentMatching, solveGA
from momentmatching_SQP import MomentMatchingSQP
from calculatemoments import mean, std, skewness, kurtosis, correlation

num_azioni = 4
num_scenarios = 6
exp_mean = [4.33, 5.91, 7.61, 8.09]
exp_std = [0.94, 0.82, 13.38, 15.70]
exp_skew = [0.80, 0.49, -0.75, -0.74]
exp_kur = [2.62, 2.39, 2.93, 2.97]

exp_cor = [[1, 0.6, -0.20, -0.10],
           [0.6, 1, -0.30, -0.20],
           [-0.20, -0.30, 1, 0.60],
           [-0.10, -0.20, -0.60, 1]]
size_pop = 500

log_name = os.path.join(
        '.', 'logs',
        f"{os.path.basename(__file__)[:-3]}.log"
    )
logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt="%H:%M:%S",
        filemode='w'
    )

# Initialize problem
problemGA = MomentMatching(num_azioni, num_scenarios, exp_mean, exp_std, exp_skew, exp_kur, exp_cor)

init_sol = solveGA(problemGA, size_pop)

problemSQP = MomentMatchingSQP(num_azioni, num_scenarios, exp_mean, exp_std, exp_skew, exp_kur, exp_cor)

solution = problemSQP.solve(init_sol)
funvalue = solution.fun
logging.info(f"Optimal objective function value: {funvalue}")
p_res = solution.x[:num_scenarios]
x_res = solution.x[num_scenarios:]
x_mat = x_res.reshape((num_azioni, num_scenarios))

# Evaluate solution
m = mean(x_mat, p_res)
dev = std(x_mat, p_res)
sk = skewness(x_mat, p_res)
k = kurtosis(x_mat, p_res)
cor = correlation(x_mat, p_res)
logging.info(f"Resulting probabilities:\n{p_res}")
logging.info(f"Prices:\n{x_mat}")
logging.info(f"Effective mean:\n{m}")
logging.info(f"Expected mean - effective mean:\n{exp_mean-m}")
logging.info(f"Effective Std:\n{dev}")
logging.info(f"Expected std - effective std:\n{exp_std-dev}")
logging.info(f"Effective Skewness:\n{sk}")
logging.info(f"Expected skew - effective skew:\n{exp_skew-sk}")
logging.info(f"Effective Kurtosis:\n{k}")
logging.info(f"Expected kurt - effective kurt:\n{exp_kur-k}")
logging.info(f"Effective Correlation:\n{cor}")
logging.info(f"Expected cor - effective cor:\n{exp_cor-cor}")





