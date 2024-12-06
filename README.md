# Scenario Tree

The files contain:
- *scenarioTree* is the main class to model a scenario tree.

- *stochModel* is an abstract class that must be used to implement the model simulating the values used by the class *ScenarioTree*.

- *MomentMatching* is an example of *stochModel* which generates scenario so that the moments are close to the empirical ones. For the moment it is for financial applications.

- *checkarbitrage* contains a function checking for arbitrage (useful for financial applications).

- *calculatemoments* containt a functions computing the moments

- *brownianMotion* contains the prototype of a *stochModel* implementing a Brownian Motion.


## How to use it

Clone the directory and add it as a module in your python project.

~~~python
import numpy as np
from scenario_tree import *


class EasyStochasticModel(StochModel):
    def __init__(self, sim_setting):
        self.averages = sim_setting['averages']
        self.dim_obs = len(sim_setting['averages'])

    def simulate_one_time_step(self, parent_node, n_children):
        probs = np.ones(n_children)/n_children
        obs = np.random.multivariate_normal(
            mean=self.averages,
            cov=np.identity(self.dim_obs),
            size=n_children
        ).T # obs.shape = (len_vector, n_children)
        return probs, obs 

sim_setting = {
    'averages': [1,1,1]
}
easy_model = EasyStochasticModel(sim_setting)
scen_tree = ScenarioTree(
    name="std_MC_two_stage_tree",
    branching_factors=[100],
    len_vector=3,
    initial_value=[1,2,3],
    stoch_model=easy_model,
)
~~~
