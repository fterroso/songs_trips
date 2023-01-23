import io
import json
import os
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class StaticSongDatasetLoader(object):

    def __init__(self, source, target_song):
        self.target_song = target_song
        self.source= source
        self._read_web_data()

    def _read_web_data(self):
        with open(os.path.join('data', 'gnn_datasets', f'{self.source}', f'dataset_{self.target_song}_static_v2.json')) as f:
                self._dataset = json.load(f) 

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]) #np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        stacked_features = np.array(self._dataset["FX"])
        self.features = [
            stacked_features[i : i + self.lags, :].T
            for i in range(stacked_features.shape[0] - self.lags)
        ]
        stacked_target = np.array(self._dataset["y"])
        self.targets = [stacked_target[i + self.lags, :].T
                         for i in range(stacked_target.shape[0] - self.lags)]

    def get_dataset(self, lags=4) -> StaticGraphTemporalSignal:
        """Returning the Songs' propagation static graph iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The Chickenpox Hungary dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset