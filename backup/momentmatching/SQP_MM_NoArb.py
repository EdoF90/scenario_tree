import numpy as np
import scipy
from scipy import optimize
from .calculatemoments import mean, std, skewness, kurtosis, correlation


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
        nu = y[self.num_scenarios:2*self.num_scenarios]
        x = y[2*self.num_scenarios:]
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
        nu = y[self.num_scenarios:2*self.num_scenarios]
        x = y[2*self.num_scenarios:]
        x_matrix = x.reshape((self.num_azioni, self.num_scenarios))
        constraints = []
        constraints.append(np.sum(p) - 1)
        for i in range(self.num_azioni):
            constr_sum = 0
            for s in range(self.num_scenarios):
                constr_sum+= nu[s]*(1 + x_matrix[i,s])

            constraints.append(constr_sum - 1)

        return constraints
    
    def gen_initsol(self):
        initial_solution_parts = []
        p_init = 1 / self.num_scenarios * np.ones(self.num_scenarios)
        initial_solution_parts.append(p_init)
        nu_init = 1 / self.num_scenarios * np.ones(self.num_scenarios)
        initial_solution_parts.append(nu_init)
        for a in range(self.num_azioni):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.num_scenarios))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)
        return initial_solution
    
    def solve(self, initial_solution):
        # Define bounds
        # Define bounds
        bounds_p_nu = [(0, np.inf)] * (2 * self.num_scenarios)  # Limiti per p e nu
        bounds_x = [(None, None)] * (self.num_azioni * self.num_scenarios)  # Nessun limite per x
        bounds = bounds_p_nu + bounds_x


        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
        p_res = res.x[:self.num_scenarios]
        nu_res = res.x[self.num_scenarios:2*self.num_scenarios]
        x_res = res.x[2*self.num_scenarios:]
        x_mat = x_res.reshape((self.num_azioni, self.num_scenarios))
        return p_res, x_mat, nu_res