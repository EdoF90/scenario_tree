from .stochModel import StochModel

class BrownianMotion(StochModel):

    def __init__(self, sim_setting):
        super().__init__(sim_setting)

    def simulate_one_time_step(self, n_children):
        # return
        pass
