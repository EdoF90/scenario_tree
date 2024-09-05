# -*- coding: utf-8 -*-
from .stochModel import StochModel
import yfinance as yf
from scipy import stats
import numpy as np
from .checkarbitrage import check_arbitrage_prices
import logging

class BrownianMotionForHedging(StochModel):

    def __init__(self, sim_setting, option_list, dt, mu, sigma, rho, rnd_state): #TODO: sim_setting è inutile 
        super().__init__(sim_setting)
        self.dt = dt 
        '''self.estimate_from_data(
            sim_setting["start"],
            sim_setting["end"]
            )'''
        self.n_options = len(option_list)
        self.option_list = option_list
        self.mu = mu
        self.sigma = sigma
        self.corr = rho 
        self.rnd_state = rnd_state
    
    '''
    def estimate_from_data(self, start, end):
        hist_prices = yf.download(
            self.tickers,
            start = start,
            end = end
        )['Adj Close']
        log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
        self.mu = log_returns.mean().values
        self.sigma = log_returns.std().values
        self.corr = log_returns.corr().values'''

    def simulate_one_time_step(self, n_children, parent_node, remaining_times): # trova i prezzi di ogni stock (e delle opzioni)
                                                            # per un nodo genitore, per tutti i suoi nodi figli
        
        parent_stock_prices= parent_node[1:self.n_shares+1] 
        parent_cash_price = parent_node[0]

        arb = True
        counter = 0
        while (arb == True) and (counter<100):
            counter += 1
            if self.n_shares > 1:
                B = self.rnd_state.multivariate_normal(
                    mean = np.zeros(self.n_shares),
                    cov  = self.corr,
                    size = n_children, #TODO: ricontrollare la dimensione (e confrontare con quello di Giovanni)
                    ).T    
            else: 
                B = self.rnd_state.normal(loc = 0, scale = 1, size=n_children)
        
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
            c = self.mu - 0.5 * self.sigma**2
            Increment = np.array(c).reshape(-1,1) * self.dt + Y

            stock_prices = np.zeros((self.n_shares, n_children))
            for i in range(self.n_shares):
                for s in range(n_children):
                    stock_prices[i,s] = parent_stock_prices[i] * np.exp(Increment[i,s]) 

            arb = check_arbitrage_prices(stock_prices, parent_stock_prices)
            if (arb == False):
                logging.info(f"No arbitrage solution found after {counter} iteration(s)")
            
        if counter >= 100:
            raise RuntimeError(f"No arbitrage solution NOT found after {counter} iteration(s)")
        else:
            probs = 1/n_children * np.ones(n_children) # TODO probabilità uniformi??????
        
        # Options 
        option_prices = np.zeros((self.n_shares, n_children))
        for j in range(self.n_shares):
            S0 = stock_prices[j,:]
            time_to_maturity = remaining_times * self.dt 
            # TODO: diamo per scontato che siano tutte opzioni europee, così da poter usare B&S
            option_prices[j,:] = self.option_list[j].BlackScholesPrice(S0, time_to_maturity)


        # Cash 
        cash_price = parent_cash_price * np.exp(self.option_list[0].risk_free_rate*self.dt) * np.ones(shape=n_children)

        prices = np.vstack((cash_price, stock_prices, option_prices))
        #TODO: inserire nel nostro main
        return probs, prices
    

    '''
    def generate_states(self):
        size = (self.branching_factor,)  #((self.n_underlyings, self.branching_factor))
        if self.dynamics == 'BS':
            if self.n_shares > 1:
                B = random.multivariate_normal(
                    mean = np.zeros(self.n_underlyings),
                    cov  = self.rho,
                    size = size,  #n_underlyings is automatic
                    ).T
            else: B = self.Obj.normal(loc = 0, scale = 1, size=size )
            Y = np.array(self.sigma).reshape(-1,1) * np.sqrt(self.dt) * B
            # c_rn   = self.r - 0.5 * self.sigma**2
            c_hist = self.mu - 0.5 * self.sigma**2
        if self.dynamics == 'VG':
            G = self.Obj.gamma( shape = self.dt/self.nu, scale = self.nu, size=size ) # scale = 1 / rate
            Y = self.Obj.normal( loc = self.mu*G, scale = self.sigma*np.sqrt(G), size=size )
            # c_rn   = self.r + np.log(1 - self.nu*self.mu - self.nu*self.sigma**2/2) / self.nu        
            c_hist = self.c
        # c = c_rn  #-> if we want a risk-neutral 'c'
        c = c_hist
        Inc = np.array(c).reshape(-1,1) * self.dt + Y
        n_u = self.n_underlyings
        self.next_states[1:n_u+1, :] = self.current_state[1:n_u+1].reshape(-1,1) * np.exp( Inc )
        '''
