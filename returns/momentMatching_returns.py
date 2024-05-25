import yfinance as yf
from ..stochModel import StochModel
import numpy as np
from scipy import stats
from scipy import optimize
from ..calculatemoments import mean, std, skewness, kurtosis, correlation
from .checkarbitrage_returns import check_arbitrage_prices
import logging
import os

log_name = os.path.join(
        '.', 'logs',
        f"momentMatching_returns.log"
    )
logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt="%H:%M:%S",
        filemode='w'
    )

class MomentMatching(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.n_children = 0
        self.ExpectedMomentsEstimate()
    
    def ExpectedMomentsEstimate(self):
        data = yf.download(self.tickers, start='2023-05-01', end='2024-05-01')['Adj Close']
        returns = data.pct_change().dropna()

        self.exp_mean = returns.mean().values
        self.exp_std = returns.std().values
        self.exp_skew = returns.apply(lambda x: stats.skew(x, bias=False)).values
        self.exp_kur = returns.apply(lambda x: stats.kurtosis(x, bias=False, fisher=False)).values
        self.exp_cor = returns.corr().values
    
    '''
    def update_expectedstd(self, parent_node):
        for i in range(self.n_shares):
            self.exp_std[i] = self.VC[i] * abs(parent_node[i] - self.exp_mean[i]) + (1-self.VC[i]) * (self.exp_std[i])
    
    def update_expectedmean(self, parent_node):
        for i in range(self.n_shares):
            self.exp_mean[i] = self.risk_free_return + self.RP[i] * self.exp_std[i]
    '''
    def get_n_children(self, n_children):
        self.n_children = n_children
    
    def _objective(self, y):
        p = y[:self.n_children]
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
        return np.sum(p) - 1
    
    
    def solve(self):
        # Define initial solution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds_p= [(0, np.inf)] * (self.n_children) 
        bounds_x = [(None, None)] * (self.n_shares * self.n_children)
        bounds = bounds_p + bounds_x
        

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        p_res = res.x[:self.n_children]
        x_res = res.x[self.n_children:]
        x_mat = x_res.reshape((self.n_shares, self.n_children))
        return p_res, x_mat, res.fun

    def simulate_one_time_step(self, n_children, parent_node, period):
        '''
        if period > 1:
            self.update_expectedstd(parent_node)
            self.update_expectedmean(parent_node)
        '''        
        self.get_n_children(n_children)
        arb = True
        counter = 0
        while arb and (counter < 100):
            probs, returns, fun = self.solve()
            counter += 1
            arb = check_arbitrage_prices(returns)
            if arb == False:
                tree_mean = mean(returns, probs)
                tree_std = std(returns, probs)
                tree_skew = skewness(returns, probs)
                tree_kurt = kurtosis(returns, probs)
                tree_cor = correlation(returns, probs)
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
                logging.info(f"Objective function value: {fun}")
                logging.info(f"Expected vs Generated Moments:")
                logging.info(f"Mean: Expected = {self.exp_mean}, Generated = {tree_mean}, Exp-Gen = {self.exp_mean - tree_mean}")
                logging.info(f"Std Dev: Expected = {self.exp_std}, Generated = {tree_std}, Exp-Gen = {self.exp_std - tree_std}")
                logging.info(f"Skewness: Expected = {self.exp_skew}, Generated = {tree_skew}, Exp-Gen = {self.exp_skew - tree_skew}")
                logging.info(f"Kurtosis: Expected = {self.exp_kur}, Generated = {tree_kurt}, Exp-Gen = {self.exp_kur - tree_kurt}")
                logging.info(f"Correlation: Expected = {self.exp_cor}, Generated = {tree_cor}, Exp-Gen = {self.exp_cor - tree_cor}")

        if counter >= 100:
            raise RuntimeError("Arbitrage-free scenarios not found")
        else:
            return probs, returns
                
    #TODO: Need to properly define this function
    def simulate_all_horizon(self, time_horizon):

        return 0