# -*- coding: utf-8 -*-
import numpy as np
import gurobipy as grb

def check_arbitrage_prices(new_prices, prev_prices):
    num_scenarios = new_prices.shape[1]
    num_assets = new_prices.shape[0]
    
    model = grb.Model()
    model.setParam('OutputFlag', 0)
    
    pi = model.addVars(num_scenarios, lb = 0, name="pi")
    
    
    for k in range(num_assets):
        model.addConstr(
            grb.quicksum(pi[n] * new_prices[k,n] for n in range(num_scenarios)) == prev_prices[k],
            f"eq1_{k}"
        )
    
    model.setObjective(0)
    
    model.optimize()
    
    if model.status == grb.GRB.Status.OPTIMAL:
        return False # Il sistema ha una soluzione, quindi non c'è arbitraggio
    else:
        return True # Il sistema non ha una soluzione, quindi c'è arbitraggio
