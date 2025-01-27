import logging
import time
import numpy as np
from scipy import optimize
from .stochModel import StochModel
from .checkarbitrage import check_arbitrage_prices
from .calculatemoments import mean, std, skewness, kurtosis, correlation


class MomentMatchingForHedging(StochModel): # Instance of the abstract class StochModel

    ''' 
    Stochastic model used in hedging settings to simulate stock price 
    dynamics and children probabilities by matching simulated properties 
    (first, third and fourth moments, std, correlation,...) with 
    historical ones. Options are priced via Black&Scholes formula.
    '''

    def __init__(self, 
                 tickers, 
                 option_list,
                 dt, r,  mu, sigma, 
                 rho, skew, kur,
                 rnd_state):
        self.n_shares = len(tickers)
        self.n_options = len(option_list)
        self.option_list = option_list
        self.dt = dt
        self.risk_free_rate = r
        self.mu = mu * self.dt 
        self.sigma = sigma * np.sqrt(self.dt)
        self.corr = rho
        self.skew = skew 
        self.kur = kur 
        self.rnd_state = rnd_state
    

    def _objective(self, y): 
        '''Objective function of the MM model. y is the vector of decision variables: 
        the first n_children entries are node probabilities, the remaining are log-returns.'''

        p = y[:self.n_children] # probabilities
        x = y[self.n_children:] # log-returns
        # log-returns in matrix form (rows: shares , columns: scenarios)
        x_matrix = x.reshape((self.n_shares, self.n_children)) 
        # Following lines calculate the statistical moments of the tree
        tree_mean = mean(x_matrix, p)
        tree_std = std(x_matrix, p) 
        tree_skew = skewness(x_matrix, p)
        tree_kurt = kurtosis(x_matrix, p)
        tree_cor = correlation(x_matrix, p)

        # The objective function is the sum of the squared difference between each expected moment
        # and the moment of the generated tree
        sqdiff = (np.linalg.norm(self.mu - tree_mean, 2) + np.linalg.norm(self.sigma - tree_std, 2) + 
                np.linalg.norm(self.skew - tree_skew, 2) + #+ np.linalg.norm(self.kur - tree_kurt, 2) +
                np.linalg.norm(self.corr - tree_cor, 1))
        
        return sqdiff
    
    
    def _constraint(self, y):
        '''Probs sum up to one.'''

        p = y[:self.n_children]
        return np.sum(p) - 1
    

    def solve(self):
        '''Solve an SLSQP problem to find probabilities and values.'''

        # Define an initial solution: uniform nodes probabilities, log-returns 
        # sampled from a Normal distribution
        initial_solution_parts = []
        p_init = (1 / self.n_children) * np.ones(self.n_children)
        initial_solution_parts.append(p_init)
        for a in range(self.n_shares):
            part = np.abs(self.rnd_state.normal(loc=self.mu[a], scale=self.sigma[a], size=self.n_children))
            initial_solution_parts.append(part)
                
        initial_solution = np.concatenate(initial_solution_parts)

        # Define probabilities bounds 
        bounds_p= [(0, 1)] * (self.n_children) 

        # Define log-returns bounds
        bounds_x = [(None, None)] * (self.n_shares * self.n_children) 

        bounds = bounds_p + bounds_x
        
        # Define constraints
        constraints = [{'type': 'eq', 'fun': self._constraint}]

        # Run optimization 
        res = optimize.minimize(self._objective, initial_solution, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 5000})
        
        # Store the solution
        p_res = res.x[:self.n_children]  # probabilities
        x_mat = res.x[self.n_children:].reshape((self.n_shares, self.n_children)) # values

        return p_res, x_mat, res.fun
    

    def simulate_one_time_step(self, n_children, parent_node):

        ''' 
        It generates children nodes by computing new asset values and 
        the probabilities of each new node. Stock prices are generated 
        until a no arbitrage setting is found. If the market is arbitrage 
        free, option prices (using Black and Scholes formula) and cash 
        new values are computed.
        '''
        
        remaining_times = parent_node['remaining_times']-1 # remaining times of children nodes
        parent_stock_prices= parent_node['obs'][1:self.n_shares+1]  
        parent_cash_price = parent_node['obs'][0]

        self.n_children = n_children # set the number of nodes to generate
        
        # Inizialization
        arb = True
        counter = 0
        flag = True
        start_t = time.time()

        # Main loop: keeps solving the MM model until an arbitrage-free good quality 
        # solution is found (or until the maximum number of iterations is reached)
        while flag and (counter < 100):
            counter += 1

            probs, returns, fun = self.solve() # solve the MM model

            # Get prices from returns
            stock_prices = np.zeros((len(parent_stock_prices), n_children))
            for i in range(len(parent_stock_prices)):
                for s in range(n_children):
                    stock_prices[i,s] = parent_stock_prices[i] * np.exp(returns[i,s])
            
            # Check if there is an arbitrage opportunity
            arb = check_arbitrage_prices(stock_prices, parent_stock_prices)

            # If an arbitrage-free and good quality solution is found, then log some 
            # info, add the generated nodes to the tree and break the loop
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
                logging.info(f"Mean: Expected = {self.mu}, Generated = {tree_mean}, Exp-Gen = {self.mu - tree_mean}")
                logging.info(f"Std Dev: Expected = {self.sigma}, Generated = {tree_std}, Exp-Gen = {self.sigma - tree_std}")
                logging.info(f"Skewness: Expected = {self.skew}, Generated = {tree_skew}, Exp-Gen = {self.skew - tree_skew}")
                #logging.info(f"Kurtosis: Expected = {self.kur}, Generated = {tree_kurt}, Exp-Gen = {self.kur - tree_kurt}")
                logging.info(f"Correlation: Expected = {self.corr}, Generated = {tree_cor}, Exp-Gen = {self.corr - tree_cor}")

        if counter >= 100:
            raise RuntimeError("Good quality arbitrage-free scenarios not found after 100 attempts")
        else:
            end_t = time.time()
            logging.info(f"Computational time to build the tree:{end_t - start_t} seconds")
        
        # Options values 
        option_prices = np.zeros((self.n_options, n_children))
        time_to_maturity = remaining_times * self.dt 
        if time_to_maturity != 0:
            for j, option in enumerate(self.option_list):
                underlying_value = stock_prices[option.underlying_index,:]
                option_prices[j,:] = option.BlackScholesPrice(underlying_value, time_to_maturity)
        else: 
            for j, option in enumerate(self.option_list):
                underlying_value = stock_prices[option.underlying_index,:]
                option_prices[j,:] = option.get_payoff(underlying_value)

        # Cash value
        cash_price = parent_cash_price * np.exp(self.risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        
        return probs, prices # return probabilities and prices to add nodes to the tree
