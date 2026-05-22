import time
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from typing import Sequence, Dict, Tuple, List
from tqdm import tqdm

import numpy as np
import cv2

from google import genai
from google.genai import types

from ..perception.segmentation.SegmentationModel import SegmentationModel
from ..mapping.ObjectAffordances import Affordance, ObjectAffordances


log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# JSON schema for bounding‐box output from Gemini
# ----------------------------------------------------------------------------
BBOX_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {
                "type": "array",
                "description": "Normalized [ymin, xmin, ymax, xmax] in 0–1000",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "label": {"type": "string"},
        },
        "required": ["box_2d", "label"],
        "propertyOrdering": ["box_2d", "label"],
    },
}


class GeminiPointer:
    """Zero-shot bbox->SAM->3D-mask pipeline using Gemini."""

    _palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (0, 128, 255),
    ]

    def __init__(
        self,
        model: str,
        system_prompt: str,
        segmentation_model: SegmentationModel,
        vis: bool,
        vis_dir: str,
        thinking_budget: int = 1024,
        max_images: int = 3,
        bg_mask_mode: str = "no_mask",
        debug: bool = False,
        debug_dir: str = ".",
    ):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.system_prompt = system_prompt
        self.thinking_budget = thinking_budget
        self.segmentation_model = segmentation_model
        self.max_images = max_images
        self.bg_mask_mode = bg_mask_mode
        self.vis = vis
        self.vis_dir = vis_dir

        # shared config for every call
        if self.model == "gemini-1.5-pro":
            self._base_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=BBOX_SCHEMA,
            )
        else:
            self._base_config = types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
                response_mime_type="application/json",
                response_schema=BBOX_SCHEMA,
            )

        # Debug stuff
        self.debug = debug
        self.debug_dir = Path(debug_dir) / "gemini_pointer"
        self.debug_counter = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            log.info(f"Debug directory created at {self.debug_dir}")

    def annotate_map(self, objects: Sequence, skip_aff_types: List[str] = []) -> None:
        pbar = tqdm(objects, desc="Affordance annotation")
        for obj in pbar:
            segs = obj.segments.get_sorted()
            views = [s.rgb for s in segs][: self.max_images]
            masks = [s.mask for s in segs][: self.max_images]
            point_maps = [s.point_map for s in segs][: self.max_images]
            prep_views = [self._apply_bg_mask(v, m) for v, m in zip(views, masks)]

            # Clear affordances using the new container.
            obj.affordance_instances = ObjectAffordances()
            inst_id_gen = iter(range(10_000_000))

            for aff_tag, aff_type in zip(obj.affordance_tags, obj.affordance_types):
                if aff_type in skip_aff_types:
                    log.info("Skipping box prediction for affordance type %s", aff_type)
                    pcd = np.array(obj.pcd.points)
                    inst_id = next(inst_id_gen)
                    obj.affordance_instances[inst_id] = Affordance(
                        tag=aff_tag,
                        type=aff_type,
                        interaction_points=[pcd.mean(axis=0).tolist()],
                        mask_3d=pcd,  # whole object
                        score=0.0,
                    )
                    continue

                save_root = os.path.join(self.vis_dir, aff_type.replace(" ", "_"))
                os.makedirs(save_root, exist_ok=True)

                for vi, img in enumerate(prep_views):
                    # 1) get zero-shot boxes from Gemini
                    boxes = self._predict_boxes(img, obj.tag, aff_tag)
                    if not boxes:
                        continue

                    masks_per_inst = {}
                    boxes_per_inst = {}

                    # 2) box -> SAM -> mask -> 3D mask + center point
                    for box_obj in boxes:
                        box_norm = box_obj["box_2d"]
                        box_px = self._norm_box_to_pixels(
                            box_norm, img.shape[1], img.shape[0]
                        )

                        # run SAM wrapper, to get np array mask and score
                        masks2d, scores = self.segmentation_model.segment_affordances(
                            img, box=box_px
                        )

                        mask2d = masks2d[0] if len(masks2d) > 0 else None
                        score = round(float(scores[0]), 3) if len(scores) > 0 else None

                        if mask2d is None or not mask2d.any():
                            continue

                        # build the 3D mask array
                        pm = point_maps[vi]  # HxWx3
                        coords2d = np.argwhere(mask2d)  # Nx2 of (row, col)
                        mask3d = np.array([pm[r, c] for r, c in coords2d], dtype=float)
                        if mask3d.size == 0:
                            continue

                        # get center3d as the mean of all points in mask3d
                        center3d = tuple(mask3d.mean(axis=0).tolist())

                        inst_id = next(inst_id_gen)
                        masks_per_inst[inst_id] = mask2d
                        boxes_per_inst[inst_id] = box_px

                        obj.affordance_instances[inst_id] = Affordance(
                            tag=aff_tag,
                            type=aff_type,
                            interaction_points=[center3d],
                            mask_3d=mask3d,
                            score=score,
                        )

                    # 3) draw and save this view
                    if self.vis:
                        self._save_intermediate_visualization(
                            img,
                            boxes_per_inst,
                            masks_per_inst,
                            vi,
                            save_root,
                            obj.id,
                            obj.tag,
                            aff_tag,
                        )

            obj.align_and_merge_affordance_instances()

    def _predict_boxes(
        self, img: np.ndarray, obj: str, aff: str
    ) -> List[Dict[str, object]]:
        prompt = (
            f"Output a JSON list of bounding boxes (keys 'box_2d' and 'label') for affordance '{aff}' on the object '{obj}'. "
            f"box_2d should be [ymin, xmin, ymax, xmax] normalized to 0–1000."
            # f"If you think the mentioned affordance is not present, return an empty list." # Moved to system prompt
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(
                data=cv2.imencode(".jpg", img)[1].tobytes(),
                mime_type="image/jpeg",
            ),
        ]

        for attempt in range(10):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=self._base_config,
                )
            except Exception as e:
                log.error(
                    f"Attempt {attempt + 1}: Failed for GeminiPointer with error: {e}"
                )
                time.sleep(0.1 * 2**attempt)
                continue
            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                try:
                    parsed = json.loads(resp.text)
                except Exception:
                    log.error(
                        f"Attempt {attempt + 1}: Failed to parse boxes JSON: %s",
                        getattr(resp, "text", "{}"),
                    )
                    continue
            break
        else:
            log.error("All attempts failed in _predict_boxes, returning empty list.")
            return []

        if self.debug:
            # Dump prompt and response to debug directory
            debug_file = self.debug_dir / f"debug_{self.debug_counter:04d}.json"
            with open(debug_file, "w") as f:
                json.dump(
                    {
                        "system_prompt": self.system_prompt,
                        "prompt": prompt,
                        "response": parsed,
                    },
                    f,
                    indent=2,
                )
            log.info(f"Debug output saved to {debug_file}")
            self.debug_counter += 1

        return parsed  # List[{"box_2d":[…],"label":…}]

    @staticmethod
    def _norm_box_to_pixels(
        norm_box: List[int], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        y0, x0, y1, x1 = norm_box
        x_min = int(x0 / 1000 * width)
        y_min = int(y0 / 1000 * height)
        x_max = int(x1 / 1000 * width)
        y_max = int(y1 / 1000 * height)
        return x_min, y_min, x_max, y_max

    def _save_intermediate_visualization(
        self,
        img: np.ndarray,
        boxes: Dict[int, Tuple[int, int, int, int]],
        masks: Dict[int, np.ndarray],
        view_idx: int,
        save_root: str,
        obj_id: str,
        obj_tag: str,
        aff_tag: str,
    ) -> None:
        if not boxes and not masks:
            return
        vis = img.copy().astype(np.uint8)
        overlay = vis.copy()

        # color in masks
        for inst_id, m in masks.items():
            color = self._palette[inst_id % len(self._palette)]
            overlay[m] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

        # draw boxes
        for inst_id, (x0, y0, x1, y1) in boxes.items():
            color = self._palette[inst_id % len(self._palette)]
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

        # Save visualization with updated naming convention
        out_path = os.path.join(
            save_root, f"{obj_tag}_{aff_tag}_view{view_idx}_{obj_id}.png"
        )
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _apply_bg_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.bg_mask_mode == "no_mask":
            return img
        fg = mask == 2
        if self.bg_mask_mode == "black":
            return np.where(fg[:, :, None], img, 0)
        if self.bg_mask_mode == "white":
            return np.where(fg[:, :, None], img, 255)
        if self.bg_mask_mode == "greyscale":
            grey = np.mean(img, 2, keepdims=True).astype(np.uint8)
            return np.where(fg[:, :, None], img, grey)
        return img

    def annotate_map_with_descs(
        self, objects: Sequence, descs: List[Tuple[str, str]]
    ) -> None:

        pbar = tqdm(objects, desc="Affordance annotation")
        for obj in pbar:
            segs = obj.segments.get_sorted()
            views = [s.rgb for s in segs][: self.max_images]
            masks = [s.mask for s in segs][: self.max_images]
            point_maps = [s.point_map for s in segs][: self.max_images]
            prep_views = [self._apply_bg_mask(v, m) for v, m in zip(views, masks)]

            # Clear affordances using the new container.
            obj.affordance_instances = ObjectAffordances()
            inst_id_gen = iter(range(10_000_000))

            if len(obj.affordance_info_with_desc_ids) == 0:
                continue

            for affordance_info in obj.affordance_info_with_desc_ids:

                desc_id = affordance_info[0]
                obj.affordance_tags = affordance_info[1]
                obj.affordance_types = affordance_info[2]

                for aff_tag, aff_type in zip(obj.affordance_tags, obj.affordance_types):
                    save_root = os.path.join(self.vis_dir, aff_type.replace(" ", "_"))
                    os.makedirs(save_root, exist_ok=True)

                    for vi, img in enumerate(prep_views):
                        # 1) get zero-shot boxes from Gemini
                        boxes = self._predict_boxes(img, obj.tag, aff_tag)
                        if not boxes:
                            continue

                        masks_per_inst = {}
                        boxes_per_inst = {}

                        # 2) box -> SAM -> mask -> 3D mask + center point
                        for box_obj in boxes:
                            box_norm = box_obj["box_2d"]
                            box_px = self._norm_box_to_pixels(
                                box_norm, img.shape[1], img.shape[0]
                            )

                            # run SAM wrapper, to get np array mask and score
                            masks2d, scores = (
                                self.segmentation_model.segment_affordances(
                                    img, box=box_px
                                )
                            )

                            mask2d = masks2d[0] if len(masks2d) > 0 else None
                            score = (
                                round(float(scores[0]), 3) if len(scores) > 0 else None
                            )

                            if mask2d is None or not mask2d.any():
                                continue

                            # build the 3D mask array
                            pm = point_maps[vi]  # HxWx3
                            coords2d = np.argwhere(mask2d)  # Nx2 of (row, col)
                            mask3d = np.array(
                                [pm[r, c] for r, c in coords2d], dtype=float
                            )
                            if mask3d.size == 0:
                                continue

                            # get center3d as the mean of all points in mask3d
                            center3d = tuple(mask3d.mean(axis=0).tolist())

                            inst_id = next(inst_id_gen)
                            masks_per_inst[inst_id] = mask2d
                            boxes_per_inst[inst_id] = box_px

                            obj.affordance_instances[inst_id] = Affordance(
                                tag=aff_tag,
                                type=aff_type,
                                interaction_points=[center3d],
                                mask_3d=mask3d,
                                score=score,
                                desc_id=desc_id,
                            )

                        # 3) draw and save this view
                        if self.vis:
                            self._save_intermediate_visualization(
                                img,
                                boxes_per_inst,
                                masks_per_inst,
                                vi,
                                save_root,
                                obj.id,
                                obj.tag,
                                aff_tag,
                            )

            obj.align_and_merge_affordance_instances_by_desc_id()
