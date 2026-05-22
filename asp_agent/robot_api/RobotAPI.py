import numpy as np


class RobotAPI:
    def get_rgbd_obs(self) -> dict[str, np.ndarray]:
        """Get the RGB-D observation from the robot's sensors."""
        pass

    def run_skill(
        self, skill: str, obj_pcd: np.ndarray, aff_pcd: np.ndarray
    ) -> tuple[bool, str]:
        raise NotImplementedError("run_skill is not implemented for this robot API.")
