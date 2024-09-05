# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .stochModel import StochModel
from networkx.drawing.nx_pydot import graphviz_layout


def prod(val):  
    res = 1 
    for ele in val:  
        res *= ele  
    return res   


class ScenarioTree(nx.DiGraph):
    def __init__(self, name: str, branching_factors: list, len_vector: int, initial_share_price, stoch_model: StochModel):
        nx.DiGraph.__init__(self)
        self.starting_node = 0
        self.len_vector = len_vector
        self.stoch_model = stoch_model
        self.add_node(
            self.starting_node,
            prices=initial_share_price,
            prob=1,
            id=0,
            stage=0
        )        
        self.name = name
        self.filtration = []

        self.branching_factors = branching_factors
        depth = len(branching_factors)

        self.n_scenarios = prod(self.branching_factors)

        self.nodes_time = []
        self.nodes_time.append([self.starting_node])

        count = 1
        last_added_nodes = [self.starting_node]
        for i in range(depth):
            next_level = []
            self.nodes_time.append([])
            self.filtration.append([])
            for parent_node in last_added_nodes:
                p, x = self._generate_one_time_step(self.branching_factors[i], self._node[parent_node]['prices'])
                for j in range(self.branching_factors[i]):
                    id_new_node = count
                    self.add_node(
                        id_new_node,
                        prices= x[:,j],
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

    def get_filtration_time(self, t):
        return self.filtration[t]

    def get_nodes_time(self, t):
        return self.nodes_time[t]

    def get_matrix_obs(self, t):
        # each row is a share, each column a scenario
        ris = np.zeros(shape=(
            len(self.nodes()[0]['prices']),
            len(self.nodes_time[t])
        ))
        for s, ele in enumerate(self.nodes_time[t]):
            ris[:, s] = self.nodes()[ele]['prices']
        return ris

    def get_leaves(self):
        return self.leaves
    
    def get_history_node(self, n):      
        ris = np.array([self.nodes[n]['prices']]).T
        if n == 0:
            return ris 
        while n != self.starting_node:
            n = list(self.predecessors(n))[0]
            ris = np.hstack((np.array([self.nodes[n]['prices']]).T, ris))
        return ris

    def print_matrix_form_on_file(self, name_details=""):
        f = open(
            os.path.join(
               "." ,
               "results",
               f"tree_matrix_form_{name_details}.csv",
            ),           
            "w"
        )
        f.write("leaf, share\n")
        for share in range(self.len_vector):
            for leaf in self.leaves:
                y = self.get_history_node(leaf)
                str_values = ",".join([f"{ele}" for ele in y[share,:]])
                f.write(f"{leaf},{share},{str_values}\n")
        f.close()

    def plot(self, file_path=None):
        _, ax = plt.subplots(figsize=(20, 12))
        x = np.zeros(self.n_nodes)
        y = np.zeros(self.n_nodes)
        x_spacing = 15
        y_spacing = 200000
        for time in self.nodes_time:
            for node in time:
                prices_str = ', '.join([f"{price:.2f}" for price in self.nodes[node]['prices']])
                ax.text(
                    x[node], y[node], f"[{prices_str}]", 
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

    def _generate_one_time_step(self, n_scenarios, parent_node):
        '''
        It returns a new period of the tree with correpsonding probabilities
        '''
        prob, prices = self.stoch_model.simulate_one_time_step(
            n_scenarios,
            parent_node,
        )
        return prob, prices
    
    def plot_all_scenarios(self):
        for leaf in self.leaves:    
            y = self.get_history_node(leaf)
            for share in range(self.len_vector):
                plt.plot(y[share, :], label=f'Share {share}')
            plt.legend()
            plt.ylabel(f"History scenario {leaf}")
            plt.savefig(
                os.path.join(
                    '.',
                    'results',
                    f'scenario_{leaf}.png'
                )
            )
            plt.close()
