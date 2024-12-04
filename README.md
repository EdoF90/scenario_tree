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

