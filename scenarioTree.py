# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .stochModel import StochModel


def prod(val):  
    res = 1 
    for ele in val:  
        res *= ele  
    return res   


class ScenarioTree(nx.DiGraph):
    def __init__(self, name: str, branching_factors: list, len_vector: int, initial_value, stoch_model: StochModel):
        nx.DiGraph.__init__(self)
        starttimer = time.time()
        self.starting_node = 0
        self.len_vector = len_vector # number of shares available in the market
        self.stoch_model = stoch_model # stochastic model used to generate the tree
        self.add_node(
            self.starting_node,
            obs=initial_value,
            prob=1,
            id=0,
            stage=0
        ) # add the node 0     
        self.name = name
        self.filtration = []
        self.branching_factors = branching_factors
        depth = len(branching_factors)
        self.n_scenarios = prod(self.branching_factors)
        self.nodes_time = []
        self.nodes_time.append([self.starting_node])

        # Build the tree
        count = 1
        last_added_nodes = [self.starting_node]
        # Main loop: until the time horizon is reached
        for i in range(depth):
            next_level = []
            self.nodes_time.append([])
            self.filtration.append([])
            
            # For each node of the last generated period add its children through the StochModel class
            for parent_node in last_added_nodes:
                # Probabilities and observations are given by the stochastic model chosen (MM or GBM)
                p, x = self._generate_one_time_step(self.branching_factors[i], self._node[parent_node]['obs'], remaining_times=depth-1-i)
                # Add all the generated nodes to the tree
                for j in range(self.branching_factors[i]):
                    id_new_node = count
                    self.add_node(
                        id_new_node,
                        obs= x[:,j],
                        prob= p[j],
                        id=count,
                        stage=i
                    )
                    self.add_edge(parent_node, id_new_node)
                    next_level.append(id_new_node)
                    self.nodes_time[-1].append(id_new_node)
                    count += 1
            last_added_nodes = next_level
            self.n_nodes = count
        self.leaves = last_added_nodes

        endtimer = time.time()
        logging.info(f"Computational time to generate the entire tree:{endtimer-starttimer} seconds")
    
    # Method to plot the tree
    def plot(self, file_path=None):
        _, ax = plt.subplots(figsize=(20, 12))
        x = np.zeros(self.n_nodes)
        y = np.zeros(self.n_nodes)
        x_spacing = 15
        y_spacing = 200000
        for time in self.nodes_time:
            for node in time:
                obs_str = ', '.join([f"{ele:.2f}" for ele in self.nodes[node]['obs']])
                ax.text(
                    x[node], y[node], f"[{obs_str}]", 
                    ha='center', va='center', bbox=dict(
                        facecolor='white',
                        edgecolor='black'
                    )
                )
                children = [child for parent, child in self.edges if parent == node]
                if len(children) % 2 == 0:
                    iter = 1
                    for child in children:
                        x[child] = x[node] + x_spacing
                        y[child] = y[node] + y_spacing * (0.5 * len(children) - iter) + 0.5 * y_spacing
                        ax.plot([x[node], x[child]], [y[node], y[child]], '-k')
                        prob = self.nodes[child]['prob']
                        ax.text(
                            (x[node] + x[child]) / 2, (y[node] + y[child]) / 2,
                            f"prob={prob:.2f}",
                            ha='center', va='center',
                            bbox=dict(facecolor='yellow', edgecolor='black')
                        )                        
                        iter += 1
                
                else:
                    iter = 0
                    for child in children:
                        x[child] = x[node] + x_spacing
                        y[child] = y[node] + y_spacing * ((len(children)//2) - iter)
                        ax.plot([x[node], x[child]], [y[node], y[child]], '-k')
                        prob = self.nodes[child]['prob']
                        ax.text(
                            (x[node] + x[child]) / 2, (y[node] + y[child]) / 2,
                            f"prob={prob:.2f}",
                            ha='center', va='center',
                        bbox=dict(facecolor='yellow', edgecolor='black')
                        )                        
                        iter += 1
            y_spacing = y_spacing * 0.25

        #plt.title(self.name)
        plt.axis('off')
        if file_path:
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()

    def _generate_one_time_step(self, n_scenarios, parent_node, remaining_times): 
        # Given a parent node and the number of children to generate, it returns the children with corresponding probabilities
        prob, obs = self.stoch_model.simulate_one_time_step(
            n_scenarios,
            parent_node,
            remaining_times
        )
        return prob, obs
    
