import os

import shutil
from pathlib import Path
from typing import Tuple, List
import torch
import numpy as np
from ultralytics import YOLOWorld

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="sam2")

from .SegmentationModel import SegmentationModel

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class YoloSAM2(SegmentationModel):
    def __init__(
        self,
        yolo_checkpoint_path: str,
        yolo_class_path: str,
        yolo_device: str,
        sam_config_file: str,
        sam_checkpoint_path: str,
        sam_prompting: str,
        sam_device: str,
        debug_images: bool,
        debug_dir: str = ".",
    ):
        self.yolo_checkpoint_path = yolo_checkpoint_path
        self.yolo_class_path = yolo_class_path
        self.yolo_device = yolo_device
        self.sam_config_file = sam_config_file
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_prompting = sam_prompting
        self.sam_device = sam_device
        self.debug_images = debug_images
        self.debug_dir = Path(debug_dir)
        self.debug_counter = 0

        # YOLO
        with open(self.yolo_class_path, "r") as f:
            self.classes = f.read().splitlines()
        self.yolo = YOLOWorld(self.yolo_checkpoint_path, verbose=False)
        self.yolo.set_classes(self.classes)
        self.yolo.to(self.yolo_device)

        # # Mobile SAM
        # mobile_sam = sam_model_registry[self.sam_model_type](
        #     checkpoint=self.sam_checkpoint_path
        # )
        # mobile_sam.to(device=self.sam_device)
        # mobile_sam.eval()

        sam2_model = build_sam2(
            self.sam_config_file, self.sam_checkpoint_path, device=self.sam_device
        )

        self.sam_predictor = SAM2ImagePredictor(sam2_model)

        if self.debug_images:
            os.makedirs(self.debug_dir / "detections", exist_ok=True)

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_bgr = img[..., ::-1].copy()
            yolo_output = self.yolo.predict(img_bgr, verbose=False)

            bbox = yolo_output[0].boxes.xyxy.cpu().numpy()

            if len(bbox) == 0:
                return None, None, None

            self.sam_predictor.set_image(img)
            if self.sam_prompting == "bbox":
                masks_np, iou_predictions_np, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox,
                    multimask_output=True,
                )
            elif self.sam_prompting == "bbox_center":
                bbox = np.asarray(bbox)
                bbox_center = (bbox[:, :2] + bbox[:, 2:]) // 2
                point_labels = np.ones((bbox_center.shape[0],), dtype=np.int32)
                masks_np, iou_predictions_np, _ = self.sam_predictor.predict(
                    point_coords=bbox_center,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True,
                )
            else:
                raise ValueError(f"Unknown prompting type: {self.sam_prompting}")

            if masks_np is None or len(masks_np) == 0 or masks_np.sum() == 0:
                return None, None, None

            # Convert outputs to torch tensors
            masks = torch.from_numpy(masks_np).to(self.sam_device)
            iou_predictions = torch.from_numpy(iou_predictions_np).to(self.sam_device)

            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            if len(iou_predictions.shape) == 1:
                iou_predictions = iou_predictions.unsqueeze(0)
            if len(bbox.shape) == 1:
                bbox = bbox.unsqueeze(0)

            best = torch.argmax(iou_predictions, dim=1)
            masks = masks[torch.arange(masks.size(0)), best]
            masks = masks.to(torch.bool)
            iou_predictions = iou_predictions[
                torch.arange(iou_predictions.size(0)), best
            ]
            bbox = torch.from_numpy(bbox).to(torch.int)

        if self.debug_images:
            img_name = str(self.debug_counter).zfill(7) + ".png"
            yolo_output[0].plot(
                show=False,
                save=True,
                filename=str(self.debug_dir / "detections" / img_name),
            )
            self.debug_counter += 1

        return masks, bbox, iou_predictions

    def segment_affordances(
        self,
        img: np.ndarray,
        points: List[Tuple[int, int]] = None,
        box: Tuple[int, int, int, int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        self.sam_predictor.set_image(img)

        if points is None and box is None:
            raise ValueError("Either points or box must be provided.")

        point_coords = None
        point_labels = None
        if points is not None:
            point_coords = np.array(points)
            point_labels = np.ones(len(points))

        if box is not None:
            box = np.array(box)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,  # Only return the best mask for now
        )
        masks = masks.astype(np.bool_)

        return masks, scores
