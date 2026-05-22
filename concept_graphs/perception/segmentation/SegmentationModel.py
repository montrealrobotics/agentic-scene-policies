from typing import Dict, Tuple, List

import torch
import numpy as np


class SegmentationModel:

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        img: np.ndarray
           Unprocessed uint8 RGB image of size (H, W, 3).

        Returns
        -------
        output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            masks: boolean tensor of size (N, H, W) where N is the number of masks predicted.
            bbox: int tensor of size (N, 4) following the format (x_min, y_min, x_max, y_mx).
            scores: float tensor of size (N,).

        """
        raise NotImplementedError

    def segment_affordances(
        self,
        img: np.ndarray,
        points: List[Tuple[int, int]] = None,
        box: Tuple[int, int, int, int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        img: np.ndarray
           Unprocessed uint8 RGB image of size (H, W, 3).
        points: List[Tuple[int, int]]
            List of (x, y) points to segment affordances around.
        box: Tuple[int, int, int, int]
            Bounding box around the object of interest in the format (x_min, y_min, x_max, y_max).

        Returns
        -------
        output: Tuple[np.ndarray, np.ndarray]
            masks: boolean tensor of size (N, H, W) where N is the number of masks predicted.
            scores: float tensor of size (N,).
        """
        raise NotImplementedError
