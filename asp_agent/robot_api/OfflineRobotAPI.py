import numpy as np
import torch
import logging

from rgbd_dataset.rgbd_dataset.BaseRGBDDataset import BaseRGBDDataset

from .RobotAPI import RobotAPI

log = logging.getLogger(__name__)


class OfflineRobotAPI(RobotAPI):
    def __init__(
        self, dataset: BaseRGBDDataset | None, dataloader: torch.utils.data.DataLoader
    ):
        super().__init__()
        self.dataset = dataset
        if dataset is not None and dataset != {}:
            self.dataloader = dataloader(dataset=self.dataset)
            self.dataloader_iter = iter(self.dataloader)
        else:
            self.dataloader = None
            self.dataloader_iter = None

    def get_rgbd_obs(self) -> dict[str, np.ndarray]:
        """Get the RGB-D observation from an offline rgbd_dataset."""
        if self.dataloader is None:
            raise ValueError(
                "Dataloader is not initialized. OfflineRobotAPI has been initialized without a dataset."
            )
        try:
            obs = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            obs = next(self.dataloader_iter)

        return obs

    def run_skill(
        self, skill: str, obj_pcd: np.ndarray, aff_pcd: np.ndarray
    ) -> tuple[bool, str]:
        log.info(
            f"Simulating skill execution: {skill} on object point cloud with {len(obj_pcd)} points and affordance point cloud with {len(aff_pcd)} points."
        )
        success, feedback_msg = True, ""
        return success, feedback_msg
