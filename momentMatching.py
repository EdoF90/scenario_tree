from .stochModel import StochModel

class MomentMatching(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)
        self.mu_func = sim_setting['mu_fuction']
    
    def solve():
        pass

    def simulate_one_time_step(self, n_children):
        # return
        pass
