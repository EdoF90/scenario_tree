# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from .stochModel import StochModel
from sklearn.cluster import KMeans
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

 
setseed = 42

class ScenarioTree(nx.DiGraph):
    def __init__(self, name: str, branching_factors: list, len_vector: int, initial_value, stoch_model: StochModel):
        nx.DiGraph.__init__(self)
        starttimer = time.time()
        self.starting_node = 0
        self.len_vector = len_vector # number of stochastic variables
        self.stoch_model = stoch_model # stochastic model used to generate the tree
        depth = len(branching_factors) # tree depth
        self.add_node( # add the node 0  
            self.starting_node,
            obs=initial_value,
            prob=1,
            id=0,
            stage=0,
            remaining_times=depth,
            path_prob=1 # path probability from the root node to the current node
        )    
        self.name = name
        self.filtration = []
        self.branching_factors = branching_factors
        self.n_scenarios = np.prod(self.branching_factors)
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
                # Probabilities and observations are given by the stochastic model chosen
                p, x = self._generate_one_time_step(self.branching_factors[i], self.nodes[parent_node])
                # Add all the generated nodes to the tree
                for j in range(self.branching_factors[i]):
                    id_new_node = count
                    self.add_node(
                        id_new_node,
                        obs=x[:,j],
                        prob=p[j],
                        id=count,
                        stage=i+1,
                        remaining_times=depth-1-i,
                        path_prob=p[j]*self._node[parent_node]['path_prob'] # path probability from the root node to the current node
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


    def _generate_one_time_step(self, n_scenarios, parent_node): 
        '''Given a parent node and the number of children to generate, it returns the 
        children with corresponding probabilities'''
        prob, obs = self.stoch_model.simulate_one_time_step(
            parent_node=parent_node,
            n_children=n_scenarios
        )
        return prob, obs
    
# -----------------------------------------------------------------------------

    def reduce_scenarios_kmeans_1D(self, X, mu, k, random_state=42):
        """
            Reduces a discrete 1D distribution using weighted KMeans clustering.

            Args:
                X: array of shape (N,) - original scenario values (e.g., demand)
                mu: array of shape (N,) - associated probabilities (sum = 1)
                k: int - desired number of reduced scenarios
                random_state: int - for reproducibility

            Returns:
                centers_sorted: list of the new (sorted) scenario values
                probs_sorted: list of the new (sorted) associated probabilities
        """        

        X = np.asarray(X).reshape(-1, 1)
        mu = np.asarray(mu)
        
        
        # 1) Clustering KMeans pesato
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X, sample_weight=mu)
        sse_kj = kmeans.inertia_
        
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_

        # 2) nuove probabilità come somma dei pesi in ciascun cluster
        probs = np.zeros(k)
        for i in range(len(X)):
            probs[labels[i]] += mu[i]

        # 3) Arrotonda e ordina per valore crescente della domanda
        pairs = sorted(zip(centers, probs), key=lambda x: x[0])
        centers_sorted = [round(float(c)) for c, _ in pairs]
        probs_sorted   = [round(float(p), 4) for _, p in pairs]
        
    
        return centers_sorted, probs_sorted, sse_kj
    
    def reduce_scenarios_kmeans_multiD(self, X, mu, k, random_state=42):
        """
            Reduces a discrete multi-dimensional distribution (e.g., for ATO) using weighted KMeans clustering.

            Args:
                X: array of shape (N, d) - original scenarios (e.g., [d1, d2])
                mu: array of shape (N,) - associated probabilities (sum = 1)
                k: int - desired number of reduced scenarios
                random_state: int - for reproducibility

            Returns:
                centers_sorted: list of new scenarios (sorted by d1, d2)
                probs_sorted: list of the new associated probabilities
        """

        X = np.asarray(X)
        mu = np.asarray(mu)

        # Clustering KMeans pesato
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X, sample_weight=mu)
        sse_kj = kmeans.inertia_

        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # nuove probabilità
        probs = np.zeros(k)
        for i in range(len(X)):
            probs[labels[i]] += mu[i]

        # Arrotonda e ordina i risultati per (d1, d2)
        centers_rounded = [
            [int(round(c[0])), int(round(c[1]))]
            for c in centers
        ]
        probs_rounded   = [round(float(p), 4) for p in probs]

        sorted_pairs = sorted(zip(centers_rounded, probs_rounded), key=lambda x: (x[0][0], x[0][1]))
        centers_sorted, probs_sorted = zip(*sorted_pairs)

        return list(centers_sorted), list(probs_sorted),sse_kj
    
# -----------------------------------------------------------------------------

    def reduce_scenarios_wasserstein_1D(self, X, mu, k, p=2,
                                        time_limit=None, verbose=False):
        """
            Exact selection of k scenarios that minimize the Wasserstein distance (1-D).

            Parameters
            ----------
            X          : array-like, shape (m,)   original demand points
            mu         : array-like, shape (m,)   original probabilities (sum = 1)
            k          : int                      number of scenarios to keep
            p          : int/float, default 1     L^p norm
            time_limit : int/float or None        time limit in seconds for Gurobi
            verbose    : bool                     if True, prints solver log

            Returns
            -------
            Y_sorted   : list[int]     values of the selected k scenarios
            nu_sorted  : list[float]   their probabilities (sum = 1)
        """

        X  = np.asarray(X,  dtype=float).flatten()
        mu = np.asarray(mu, dtype=float)
        m  = len(X)
        if k >= m:
            raise ValueError("k deve essere < m")

        # 1) matrice costi |xi - xj|^p
        C = self.compute_cost_matrix_unidimensional(X, X, p=p)

        # 2) modello 
        mdl = gp.Model("ScenarioReductionMIP")
        if not verbose:
            mdl.setParam("OutputFlag", 0)
        if time_limit:
            mdl.setParam("TimeLimit", time_limit)

        # variabili
        gamma = mdl.addVars(m, m, lb=0.0, name="gamma")          # continui
        z     = mdl.addVars(m, vtype=GRB.BINARY, name="z")       # binari
        nu    = mdl.addVars(m, lb=0.0, name="nu")                # continui

        # 3) obiettivo
        mdl.setObjective(gp.quicksum(C[i, j] * gamma[i, j]
                                    for i in range(m) for j in range(m)),
                        GRB.MINIMIZE)

        # 4) vincoli supply: Σ_j γ_ij = μ_i
        for i in range(m):
            mdl.addConstr(gp.quicksum(gamma[i, j] for j in range(m)) == mu[i],
                        name=f"supply_{i}")

        # 5) vincoli demand: Σ_i γ_ij = ν_j
        for j in range(m):
            mdl.addConstr(gp.quicksum(gamma[i, j] for i in range(m)) == nu[j],
                        name=f"demand_{j}")

        # 6) supporto: ν_j ≤ z_j
        for j in range(m):
            mdl.addConstr(nu[j] <= z[j], name=f"support_{j}")

        # 7) esattamente k scenari scelti
        mdl.addConstr(gp.quicksum(z[j] for j in range(m)) == k, name="cardinality")

        # 8) probabilità totali = 1
        mdl.addConstr(gp.quicksum(nu[j] for j in range(m)) == 1.0, name="sum_prob")

        # 9) solve
        mdl.optimize()

        if mdl.status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi non ha trovato ottimo (stato %s)" % mdl.Status)

        # 10) estrai risultati
        sel_idx = [j for j in range(m) if z[j].X > 0.5]
        Y       = X[sel_idx].astype(int)
        nu_vals = np.array([nu[j].X for j in sel_idx])

        # ordina
        order = np.argsort(Y)
        Y_sorted  = Y[order].tolist()
        nu_sorted = [round(float(pv), 4) for pv in nu_vals[order]]

        return Y_sorted, nu_sorted
    
    def reduce_scenarios_wasserstein_multiD(self, X, mu, k, p=2,
                                            time_limit=None, verbose=False):
        """
            Exact selection of k scenarios (multi-D) that minimize the Wasserstein p-norm distance via MILP (Gurobi).

            Parameters
            ----------
            X          : array-like, shape (m, d)   original demand vectors
            mu         : array-like, shape (m,)     original probabilities (sum = 1)
            k          : int                        number of scenarios to retain
            p          : int/float, default 2       L^p norm (2 = Euclidean)
            time_limit : int/float or None          time limit in seconds for Gurobi
            verbose    : bool                       True → print solver log

            Returns
            -------
            Y_sorted   : list[list[int]]   selected scenario vectors (sorted)
            nu_sorted  : list[float]       corresponding probabilities (sum = 1)
        """

        X  = np.asarray(X,  dtype=float)
        mu = np.asarray(mu, dtype=float).flatten()
        m, d = X.shape
        if k >= m:
            raise ValueError("k deve essere < numero scenari originali (m)")

        # 1) matrice costi ‖x_i − x_j‖_p^p
        C = self.compute_cost_matrix_multidimensional(X, X, p=p)      # shape (m, m)

        # 2) modello 
        mdl = gp.Model("ScenarioReductionMIP_multiD")
        if not verbose:
            mdl.setParam("OutputFlag", 0)
        if time_limit:
            mdl.setParam("TimeLimit", time_limit)

        gamma = mdl.addVars(m, m, lb=0.0, name="gamma")           # continue
        z     = mdl.addVars(m, vtype=GRB.BINARY, name="z")        # binarie
        nu_v  = mdl.addVars(m, lb=0.0, name="nu")                 # continue

        # 3) obiettivo
        mdl.setObjective(
            gp.quicksum(C[i, j]*gamma[i, j] for i in range(m) for j in range(m)),
            GRB.MINIMIZE)

        # 4) supply  Σ_j γ_ij = μ_i
        for i in range(m):
            mdl.addConstr(gp.quicksum(gamma[i, j] for j in range(m)) == mu[i],
                        name=f"supply_{i}")

        # 5) demand  Σ_i γ_ij = ν_j
        for j in range(m):
            mdl.addConstr(gp.quicksum(gamma[i, j] for i in range(m)) == nu_v[j],
                        name=f"demand_{j}")

        # 6) linking  ν_j ≤ z_j
        for j in range(m):
            mdl.addConstr(nu_v[j] <= z[j], name=f"support_{j}")

        # 7) cardinalità: esattamente k scenari
        mdl.addConstr(gp.quicksum(z[j] for j in range(m)) == k, name="cardinality")

        # 8) probabilità totali = 1
        mdl.addConstr(gp.quicksum(nu_v[j] for j in range(m)) == 1.0, name="sum_prob")

        # 9) solve
        mdl.optimize()
        if mdl.status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status: {mdl.Status} (non ottimale)")

        # 10) estrai scenari scelti e loro masse
        sel_idx  = [j for j in range(m) if z[j].X > 0.5]
        Y_sel    = X[sel_idx]                    # shape (k, d)
        nu_sel   = np.array([nu_v[j].X for j in sel_idx])

        # 11) ordina per comodità (prima coord 0, poi 1, …)
        order    = np.lexsort(Y_sel.T[::-1])     # ordina per colonne crescenti
        Y_sorted = Y_sel[order].round().astype(int).tolist()
        nu_sorted= [round(float(nu_sel[t]), 4) for t in order]

        return Y_sorted, nu_sorted
    
# -----------------------------------------------------------------------------

    def compute_cost_matrix_multidimensional(self, points_mu, points_nu, p=2):
        """
            Computes the cost matrix for multidimensional distributions.

            Args:
                points_mu: array of shape (m, d)
                points_nu: array of shape (n, d)
                p: norm to use (default = 2 for Euclidean distance)

            Returns:
                cost_matrix: array of shape (m, n)
        """

        m = len(points_mu)
        n = len(points_nu)

        cost_matrix = np.zeros((m, n))
        
        for i in range(m):
            for j in range(n):
                cost_matrix[i, j] = np.linalg.norm(np.array(np.array(points_mu[i]) - np.array(points_nu[j])), ord=p)
        return cost_matrix

    ###################################################################################################################################

    def compute_cost_matrix_unidimensional(self, points_mu, points_nu, p=2):
        """
        Compute the cost matrix using a given p-norm.

        Parameters:
        -----------
        points_mu : array-like, shape (m, d)
            Coordinates of points corresponding to the distribution mu (source points).
        points_nu : array-like, shape (n, d)
            Coordinates of points corresponding to the distribution nu (target points).
        p : float, optional (default=2)
            The p-norm to use for computing the cost (e.g., p=2 for Euclidean distance, p=1 for Manhattan distance).
        
        Returns:
        --------
        cost_matrix : array, shape (m, n)
            The cost matrix where cost_matrix[i, j] is the distance (cost) between points_mu[i] and points_nu[j].
        """
        m = len(points_mu)
        n = len(points_nu)
        
        cost_matrix = np.zeros((m, n))
        
        for i in range(m):
            for j in range(n):
                # Compute the p-norm distance between point i in mu and point j in nu
                cost_matrix[i, j] = abs(points_mu[i] - points_nu[j])**p
        return cost_matrix

    ###################################################################################################################################

    def aggregate_discrete_demands(self, demands, probs, round_probs=2):
        """
            Aggregates and sorts discrete scenarios by summing the probabilities associated with identical demand values.

            Args:
                demands: list of integer demand values
                probs: list of corresponding probabilities
                round_probs: number of decimal digits to round the final probabilities

            Returns:
                (demands_agg, probs_agg): parallel lists, sorted in increasing order
        """

        demand_prob = defaultdict(float)
        for d, p in zip(demands, probs):
            demand_prob[d] += p

        # Ordina per chiave (domanda crescente)
        items = sorted(demand_prob.items())

        demands_agg = [d for d, _ in items]
        probs_agg = [round(p, round_probs) for _, p in items]

        return demands_agg, probs_agg

    ###################################################################################################################################

    def aggregate_vectorial_demands(self, demand_vectors, probs, round_probs=3):
        """
            Aggregates vectorial scenarios by summing the probabilities associated with identical demand vectors,

            Args:
                demand_vectors: list of lists or tuples (e.g., [[100, 400], [80, 250], ...])
                probs: list of associated probabilities
                round_probs: number of decimal digits to round the final probabilities

            Returns:
                (demands_agg, probs_agg): sorted lists with unique demand vectors and summed probabilities
        """

        demand_prob = defaultdict(float)
        for d_vec, p in zip(demand_vectors, probs): 
            rounded_vec = tuple(d_vec)
            demand_prob[rounded_vec] += p

        # Ordina per valore dei vettori
        items = sorted(demand_prob.items(), key=lambda x: x[0])

        demands_agg = [list(k) for k, _ in items]
        probs_agg = [round(v, round_probs) for _, v in items]

        return demands_agg, probs_agg
     #Sostituito scenarioTree.py con versione aggiornata
