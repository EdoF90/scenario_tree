from .stochModel import StochModel

class ScenarioTree():
    def __init__(self, stoch_model: StochModel):
        stoch_model.simulate_one_time_step(3)

    def prova(self):
        print(33)

