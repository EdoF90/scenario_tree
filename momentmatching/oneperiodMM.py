import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.config import Config
from calculatemoments import mean, std, skewness, kurtosis, correlation

Config.warnings['not_compiled'] = False
class MomentMatching(Problem):
    def __init__(self, num_azioni, num_scenarios, exp_mean, exp_std, exp_skew, exp_kur, exp_cor):
        super().__init__(n_var=(num_azioni+1)*num_scenarios, n_obj=1, n_eq_constr=1, xl=0, xu=np.inf, type_var=float)
        self.num_azioni = num_azioni
        self.num_scenarios = num_scenarios
        self.exp_mean = exp_mean
        self.exp_std = exp_std
        self.exp_skew = exp_skew
        self.exp_kur = exp_kur
        self.exp_cor = exp_cor

    def _evaluate(self, pop, out, *args, **kwargs):
        n_pop = np.size(pop, 0)
        out["F"] = np.zeros(n_pop)
        out["H"] = np.zeros(n_pop)
        for i in range(n_pop):
            p = pop[i, 0:self.num_scenarios]
            x = pop[i, self.num_scenarios:]
            x_matrix = x.reshape((self.num_azioni, self.num_scenarios))
            tree_mean = mean(x_matrix, p)
            tree_std = std(x_matrix, p)
            tree_skew = skewness(x_matrix, p)
            tree_kurt = kurtosis(x_matrix, p)
            tree_cor = correlation(x_matrix, p)
            sqdiff = np.linalg.norm(self.exp_mean - tree_mean, 2) + np.linalg.norm(self.exp_std - tree_std, 2) + np.linalg.norm(self.exp_skew - tree_skew, 2) + np.linalg.norm(self.exp_kur - tree_kurt, 2) + np.linalg.norm(self.exp_cor - tree_cor, 1)
            out["F"][i] = sqdiff
            out["H"][i] = np.sum(p) - 1


def solveGA(problem, size_pop):
    #Define initial solution
    initial_solution_parts = []
    p_init = 1 / problem.num_scenarios * np.ones((size_pop,problem.num_scenarios))
    initial_solution_parts.append(p_init)
    for a in range(problem.num_azioni):
        part = np.abs(np.random.normal(loc=problem.exp_mean[a], scale=problem.exp_std[a], size =(size_pop, problem.num_scenarios)))
        initial_solution_parts.append(part)

    initial_solution = np.concatenate(initial_solution_parts, axis = 1)

    # Initialize algorithm
    algorithm = GA(pop_size=size_pop, sampling=initial_solution, crossover=UniformCrossover(prob=0.9), mutation=PolynomialMutation(prob=0.1),
                eliminate_duplicates=True)
    
    # Stopping criteria
    stop_criteria = DefaultSingleObjectiveTermination(
        xtol=1e-5,
        cvtol=1e-16,
        ftol=1e-5,
        period=100,
        n_max_gen=1e4,
        n_max_evals=1e6
    )
    # Running optimization
    res = minimize(problem, algorithm, termination=stop_criteria, verbose=True)
    p_res = res.X[0:problem.num_scenarios]
    x_res = res.X[problem.num_scenarios:]
    x_mat = x_res.reshape((problem.num_azioni, problem.num_scenarios))

    
    #return x_mat, p_res
    return res.X

