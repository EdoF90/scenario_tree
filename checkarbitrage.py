import numpy as np
import gurobipy as grb

def check_arbitrage_prices(returns):
    num_scenarios = returns.shape[1]
    num_assets = returns.shape[0]
    
    model = grb.Model()
    model.setParam('OutputFlag', 0)
    
    pi = model.addVars(num_scenarios, lb = 0, name="pi")
    
    
    for k in range(num_assets):
        model.addConstr(
            grb.quicksum(pi[n] * (1+returns[k,n]) for n in range(num_scenarios)) == 1,
            f"eq1_{k}"
        )
    
    model.setObjective(0)
    
    model.optimize()
    
    if model.status == grb.GRB.Status.OPTIMAL:
        return False # Il sistema ha una soluzione, quindi non c'è arbitraggio
    else:
        return True # Il sistema non ha una soluzione, quindi c'è arbitraggio
