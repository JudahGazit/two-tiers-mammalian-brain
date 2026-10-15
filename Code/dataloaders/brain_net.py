from publication_paths import data_path, output_path
import os

import networkx as nx
import numpy as np
import pandas as pd


BRAINS_LOCATIONS = data_path('brain_nets_fixed_means_knn')


class BrainNet:
    override_weights = True
    location = BRAINS_LOCATIONS

    def __init__(self, brain_name: str,
                 adjacency: np.ndarray = None,
                 distances: np.ndarray = None,
                 positions: np.ndarray = None,
                 fibers: np.ndarray = None,
                 graph: nx.Graph = None,
                 normalize_distances: bool = True,
                 location=BRAINS_LOCATIONS,
                 q=0.01):
        self.location = location
        self.brain_name = brain_name
        self.adjacency = adjacency if adjacency is not None else self._load_matrix('connections')
        self.distances = distances if distances is not None else self._load_matrix('distances')
        self.positions = positions if positions is not None else self._load_matrix('positions')
        self.fibers = fibers if fibers is not None else self._load_matrix('fibers')
        self.graph = graph if graph is not None else self._load_graph()
        self.q = q
        self._replace_zeros_with_nans()
        self._override_weights()
        if normalize_distances:
            self._normalize_distances()

    def _override_weights(self):
        if self.override_weights and self.fibers is None:
            self.fibers = self.distances * self.adjacency
            for edge in self.graph.edges:
                self.graph.edges[edge]['weight'] = self.distances[edge]

    def _replace_zeros_with_nans(self):
        if self.fibers is not None:
            self.fibers[self.fibers == 0] = np.nan
        if self.adjacency is not None:
            self.adjacency[self.adjacency == 0] = np.nan

    def _normalize_distances(self):
        d = self.distances[np.triu_indices(self.distances.shape[0], k=1)]
        clip_range = np.quantile(d, [self.q, 1 - self.q])
        self.fibers = self.fibers.clip(*clip_range)
        self.distances = self.distances.clip(*clip_range)
        self.fibers = self.fibers / np.nanmax(self.distances)
        self.distances = self.distances / np.nanmax(self.distances)

    def _load_matrix(self, file_suffix: str) -> np.ndarray:
        filename = os.path.join(self.location, self.brain_name + f'.{file_suffix}.npy')
        if os.path.isfile(filename):
            return np.array(np.load(filename, allow_pickle=True)).astype(np.float32)

    def _load_graph(self):
        if len(self.adjacency.shape) > 2:
            self.adjacency = self.adjacency[:, :, np.random.choice(self.adjacency.shape[2])]
        graph = nx.from_numpy_array(self.adjacency)
        graph = nx.convert_node_labels_to_integers(graph, ordering='sorted', label_attribute='original_label')
        return graph

    def subset(self, nodes_include=None, nodes_exclude=None):
        nodes = sorted(nodes_include) if nodes_include is not None else [
            node for node in range(self.graph.number_of_nodes()) if node not in nodes_exclude]
        subgraph_distances = self.distances[nodes][:, nodes]
        subgraph_adjacency = self.adjacency[nodes][:, nodes]
        subgraph_positions = self.positions[nodes]
        subgraph_fibers = self.fibers[nodes][:, nodes]
        subgraph_graph = self.graph.subgraph(nodes)

        dataset = BrainNet(self.brain_name, subgraph_adjacency, subgraph_distances,
                           subgraph_positions, subgraph_fibers, subgraph_graph, normalize_distances=False)
        return dataset

    def subset_edges(self, nodes_src, nodes_dst=None):
        if nodes_dst is None or np.array_equal(nodes_src, nodes_dst):
            return self.subset(nodes_src)
        is_src, is_dst = np.zeros(self.graph.number_of_nodes()), np.zeros(self.graph.number_of_nodes())
        is_src[nodes_src] = 1
        is_dst[nodes_dst] = 1
        mask = 1 - (np.multiply(*np.meshgrid(is_src, is_src)) +
                    np.multiply(*np.meshgrid(is_dst, is_dst)))
        subgraph_adjacency = np.multiply(self.adjacency, mask)
        subgraph_fibers = np.multiply(self.fibers, mask)
        subgraph_graph = nx.from_numpy_array(subgraph_adjacency)
        dataset = BrainNet(self.brain_name, subgraph_adjacency, self.distances.copy(), self.positions, subgraph_fibers, subgraph_graph)
        return dataset

    @classmethod
    def options(cls):
        if os.path.isdir(cls.location):
            all_file_names = os.listdir(cls.location)
            all_file_names = [file.split('.')[0] for file in all_file_names]
            return np.unique(all_file_names)
        return []

    @property
    def number_of_nodes(self):
        return self.adjacency.shape[0] if self.adjacency is not None else 0

    @property
    def number_of_edges(self):
        return int(np.nansum(self.adjacency) / 2) if self.adjacency is not None else 0

    @property
    def degrees(self):
        return np.nansum(self.adjacency, 0) if self.adjacency is not None else 0
