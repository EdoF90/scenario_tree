from .stochModel import StochModel
import numpy as np
import scipy
from scipy import optimize
from .calculatemoments import mean, std, skewness, kurtosis, correlation
from .checkarbitrage import check_arbitrage_prices

class MomentMatching(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.VC = sim_setting['volatilityclumping']
        self.MRF = sim_setting['meanreversionfactor']
        self.MRL = sim_setting['meanreversionfactor']
        self.RP = sim_setting['risk_premium']
        self.exp_mean = sim_setting['expectedmean']
        self.exp_std = sim_setting['expectedstd']
        self.exp_skew = sim_setting['expectedskewness']
        self.exp_kur = sim_setting['expectedkurtosis']
        self.exp_cor = sim_setting['expectedcorrelation']
        self.n_children = 0
    
    def update_expectedstd(self, parent_node):
        for i in range(self.n_shares):
            self.exp_std[i] = self.VC[i] * abs(parent_node[i] - self.exp_mean[i]) + (1-self.VC[i]) * (self.exp_std[i])
    
    def update_expectedmean(self, parent_node):
        for i in range(self.n_shares):
            if i < 2:
                self.exp_mean[i] =  self.MRF[i] * self.MRL[i] + (1-self.MRF[i]) * parent_node[i]
            else:
                self.exp_mean[i] = self.exp_mean[0] + self.RP[i-2] * self.exp_std[i]
        
    
    def get_n_children(self, n_children):
        self.n_children = n_children

    
    def _objective(self, y):
        p = y[:self.n_children]
        #nu = y[self.n_children:2*self.n_children]
        x = y[self.n_children:]
        x_matrix = x.reshape((self.n_shares, self.n_children))
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p)
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)
        sqdiff = (np.linalg.norm(self.exp_mean - tree_mean, 2) + np.linalg.norm(self.exp_std - tree_std, 2) + 
                np.linalg.norm(self.exp_skew - tree_skew, 2) + np.linalg.norm(self.exp_kur - tree_kurt, 2) +
                np.linalg.norm(self.exp_cor - tree_cor, 1))
        
        return sqdiff
    
    def _constraint(self, y):
        p = y[:self.n_children]
        #nu = y[self.n_children:2*self.n_children]
        x = y[self.n_children:]
        x_matrix = x.reshape((self.n_shares, self.n_children))
        constraints = []
        constraints.append(np.sum(p) - 1)
        '''
        for i in range(self.n_shares):
            constr_sum = 0
            for s in range(self.n_children):
                constr_sum+= nu[s]*(1 + x_matrix[i,s])

            constraints.append(constr_sum - 1)
        '''

        return constraints
    
    
    def solve(self):
        # Define initial solution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        #nu_init = (1 / self.n_children) * np.ones(self.n_children)
        #initial_solution_parts.append(nu_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds_p_nu= [(0, np.inf)] * (self.n_children)  # Limiti per p e nu
        bounds_x = [(None, None)] * (self.n_shares * self.n_children)  # Nessun limite per x
        bounds = bounds_p_nu + bounds_x

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        p_res = res.x[:self.n_children]
        #nu_res = res.x[self.n_children:2*self.n_children]
        x_res = res.x[self.n_children:]
        x_mat = x_res.reshape((self.n_shares, self.n_children))
        return p_res, x_mat, res.fun

    def simulate_one_time_step(self, n_children, parent_node, period):
        '''
        if period > 1:
            self.update_expectedmean(parent_node)
            self.update_expectedstd(parent_node)
        '''
        
        self.get_n_children(n_children)
        arb = True
        counter = 0
        while arb and (counter < 100):
            probs, returns, fun = self.solve()
            counter += 1
            arb = check_arbitrage_prices(returns)
            if arb == False:
                print(f"No arbitrage solution found after {counter} iteration")

        if counter >= 100:
            raise RuntimeError("Arbitrage-free scenarios not found")
        else:
            return probs, returns, fun
        
    #TODO: Need to properly define this function
    def simulate_all_horizon(self, time_horizon):
        return 0
