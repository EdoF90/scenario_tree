import numpy as np
import scipy
from scipy import optimize
from ..calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatchingSQP:
    def __init__(self, num_azioni, num_scenarios, exp_mean, exp_std, exp_skew, exp_kur, exp_cor):
        self.num_azioni = num_azioni
        self.num_scenarios = num_scenarios
        self.exp_mean = exp_mean
        self.exp_std = exp_std
        self.exp_skew = exp_skew
        self.exp_kur = exp_kur
        self.exp_cor = exp_cor

    def _objective(self, y):
        p = y[:self.num_scenarios]
        x = y[self.num_scenarios:]
        x_matrix = x.reshape((self.num_azioni, self.num_scenarios))
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p)
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)
        sqdiff = np.linalg.norm(self.exp_mean - tree_mean, 2) + np.linalg.norm(self.exp_std - tree_std, 2) + np.linalg.norm(self.exp_skew - tree_skew, 2) + np.linalg.norm(self.exp_kur - tree_kurt, 2) + np.linalg.norm(self.exp_cor - tree_cor, 1)
        return sqdiff
    
    def _constraint(self, y):
        p = y[:self.num_scenarios]
        return np.sum(p) - 1
    
    def gen_initsol(self):
        initial_solution_parts = []
        p_init = 1 / self.num_scenarios * np.ones(self.num_scenarios)
        initial_solution_parts.append(p_init)
        for a in range(self.num_azioni):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.num_scenarios))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)
        return initial_solution
    
    def solve(self, initial_solution):
        # Define bounds
        bounds_p= [(0, np.inf)] * ( self.num_scenarios)  # Limiti per p e nu
        bounds_x = [(None, None)] * (self.num_azioni * self.num_scenarios)  # Nessun limite per x
        bounds = bounds_p + bounds_x

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        p_res = res.x[:self.num_scenarios]
        x_res = res.x[self.num_scenarios:]
        x_mat = x_res.reshape((self.num_azioni, self.num_scenarios))
        return p_res, x_mat, res.fun
    
