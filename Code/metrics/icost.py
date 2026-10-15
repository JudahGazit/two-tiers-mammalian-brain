import random
from abc import ABC
from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse.csgraph

from dataloaders.brain_net import BrainNet
from metrics.metric import MetricBoundaries, Metric
import igraph

class ICost(ABC):
    def __init__(self, graph_dataset: BrainNet, boundaries: Union[MetricBoundaries] = None):
        self.graph_dataset = graph_dataset
        self._boundaries = boundaries or MetricBoundaries()

    def __group_by_matrix(self, data: np.ndarray):
        unique_elements, counts = np.unique(data, return_counts=True)
        data = np.array([unique_elements, counts]).transpose()
        grouped_by = pd.DataFrame(data, columns=['dist', 'count'])
        grouped_by = grouped_by[grouped_by['dist'] > 0]
        return grouped_by

    def all_path_lengths(self, weight=False) -> pd.DataFrame:
        indices = random.sample(range(self.graph_dataset.number_of_nodes), 1000) if self.graph_dataset.number_of_nodes > 1000 else None
        if self.graph_dataset.graph is not None:
            graph = igraph.Graph.from_networkx(self.graph_dataset.graph)
            all_pairs_shortest_path = graph.components().giant().shortest_paths(indices, weights='weight' if weight else None)
        else:
            all_pairs_shortest_path = scipy.sparse.csgraph.shortest_path(self.graph_dataset.adjacency, directed=False,
                                                                         unweighted=not weight, indices=indices)
        gb = self.__group_by_matrix(all_pairs_shortest_path)
        return gb

    def cost(self):
        raise NotImplementedError()

    @property
    def boundaries(self) -> MetricBoundaries:
        raise NotImplementedError()


class INeighboursCost(ICost, ABC):
    def costs_if_add(self):
        raise NotImplementedError

    def costs_if_remove(self):
        raise NotImplementedError


class RelativeCostBase(ICost, ABC):
    def relative_cost(self, subset):
        total_value = self.cost().normalized_value
        relative_boundary = MetricBoundaries(relative_value=total_value)
        relative_value = self.__class__(self.graph_dataset.subset(subset), relative_boundary).cost()
        return relative_value
