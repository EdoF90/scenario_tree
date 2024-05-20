from abc import abstractmethod

class StochModel():
    @abstractmethod
    def __init__(self, sim_setting):
        self.n_shares = sim_setting['n_shares']

    @abstractmethod
    def simulate_one_time_step(self, n_children):
        pass
