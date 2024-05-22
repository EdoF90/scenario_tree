import numpy as np
import logging
import os
from scenario_tree.backup.momentmatching.SQP_MM_NoArb import MomentMatchingSQP
from scenario_tree.calculatemoments import mean, std, skewness, kurtosis, correlation
from scenario_tree.checkarbitrage import check_arbitrage_prices

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

num_azioni = 4
num_scenarios = 6
prev_prices = np.ones(num_azioni)
exp_mean = [4.33, 5.91, 7.61, 8.09]
exp_std = [0.94, 0.82, 13.38, 15.70]
exp_skew = [0.80, 0.49, -0.75, -0.74]
exp_kur = [2.62, 2.39, 2.93, 2.97]

exp_cor = [[1, 0.6, -0.20, -0.10],
           [0.6, 1, -0.30, -0.20],
           [-0.20, -0.30, 1, 0.60],
           [-0.10, -0.20, -0.60, 1]]

# Initialize problem
problem = MomentMatchingSQP(num_azioni, num_scenarios, exp_mean, exp_std, exp_skew, exp_kur, exp_cor)
initial_solution = problem.gen_initsol()

[p_res, x_mat, nu_res] = problem.solve(initial_solution)


# Evaluate solution
m = mean(x_mat, p_res)
dev = std(x_mat, p_res)
sk = skewness(x_mat, p_res)
k = kurtosis(x_mat, p_res)
cor = correlation(x_mat, p_res)
logging.info(f"Martingale measure:\n{nu_res}")
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

[arb, pi] = check_arbitrage_prices(prev_prices, x_mat)
if arb == False:
    print("There is no arbitrage")
    print(f"{pi}")
    print(f"{nu_res}")
else:
    print("There is arbitrage")
    print(f"{pi}")
    print(f"{nu_res}")

