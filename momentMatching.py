import logging
import time
import numpy as np
import yfinance as yf
from scipy import stats
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatching(StochModel): # Instance of the abstract class StochModel

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.n_children = 0 # number of children to generate

        self.ExpectedMomentsEstimate(
            start = self.start_date,
            end = self.end_date
        ) # definition of the expected moments to match
    
    def ExpectedMomentsEstimate(self, start, end):
        hist_prices = yf.download(
            self.tickers, 
            start=start,
            end=end
            )['Close']
        monthly_prices = hist_prices.resample('ME').last()
        log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()

        self.exp_mean = log_returns.mean().values
        self.exp_std = log_returns.std().values
        self.exp_skew = log_returns.apply(
            lambda x: stats.skew(x, bias=False)
            ).values
        self.exp_kur = log_returns.apply(
            lambda x: stats.kurtosis(x, bias=False, fisher=False)
            ).values
        self.exp_cor = log_returns.corr().values
    
    def set_n_children(self, n_children): # set the number of children to generate
        self.n_children = n_children
    
    def _objective(self, y): # objective function of the MM model
        # y is the vector of decision variables: the first n_children entries are node proabilities, the remaining are log-returns
        p = y[:self.n_children] # probabilities
        x = y[self.n_children:] # log-returns
        x_matrix = x.reshape((self.n_shares, self.n_children)) # log-returns in matrix form (row: shares - column: scenario)
        # Following lines calculate the statistical moments of the tree
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p) 
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)

        # The objective function is the squared difference among the expexted moments and the moments underlying the generated tree
        sqdiff = (np.linalg.norm(self.exp_mean - tree_mean, 2) + np.linalg.norm(self.exp_std - tree_std, 2) + 
                np.linalg.norm(self.exp_skew - tree_skew, 2) + np.linalg.norm(self.exp_kur - tree_kurt, 2) +
                np.linalg.norm(self.exp_cor - tree_cor, 1))
        
        return sqdiff
    
    
    def _constraint(self, y):
        # probs sum up to one
        p = y[:self.n_children]
        return np.sum(p) - 1
    
    def solve(self):
        # Define initial solution: equal proabibilities for each node, log-returns sampled from a Normal distribution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(np.random.normal(loc=self.exp_mean[a], scale=self.exp_std[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define bounds
        bounds_p= [(0.05, 0.4)] * (self.n_children) # bounds for probabilities to avoid vanishing probabilities
        '''
        mean_bound = np.mean(self.exp_mean)
        std_bound = np.max(self.exp_std)
        lb = mean_bound - 3*std_bound
        ub = mean_bound + 3*std_bound
        bounds_x = [(lb, ub)] * (self.n_shares * self.n_children)
        '''
        bounds_x = [(None, None)] * (self.n_shares * self.n_children) # No bounds for log-returns, if you want to bound them uncomment lines 84-88

        bounds = bounds_p + bounds_x
        

        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Running optimization
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        p_res = res.x[:self.n_children]
        x_res = res.x[self.n_children:]
        x_mat = x_res.reshape((self.n_shares, self.n_children))
        return p_res, x_mat, res.fun

    def simulate_one_time_step(self, n_children, parent_node):
        self.set_n_children(n_children) # set the number of nodes to generate
        # Inizialization
        arb = True
        counter = 0
        flag = True
        startt = time.time()

        # Main loop: keeps solving the MM model until an arbitrage-free good quality solution is found (or until the maximum number of iterations is reached)
        while flag and (counter < 100):
            counter += 1

            probs, returns, fun = self.solve() # solve the MM model

            # Get prices from returns
            prices = np.zeros((len(parent_node), n_children))
            for i in range(len(parent_node)):
                for s in range(n_children):
                    prices[i,s] = parent_node[i] * np.exp(returns[i,s])
            
            # Check if there is an arbitrage opportunity
            arb = check_arbitrage_prices(prices, parent_node)
            #arb = False
            # If an arbitrage-free and good quality solution is found, then log some info, add the generated nodes to the tree and break the loop
            if (arb == False) and (fun <= 1):
                flag = False
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
            raise RuntimeError("Good quality arbitrage-free scenarios not found after 100 attempts")
        else:
            endt = time.time()
            logging.info(f"Computational time to build the tree:{endt - startt} seconds")
            return probs, prices # return probababilities and prices to add nodes to the tree
