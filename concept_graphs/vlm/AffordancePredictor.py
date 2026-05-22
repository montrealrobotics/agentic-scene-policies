import ast
from typing import List, Tuple
import numpy as np
import logging
from tqdm import tqdm
from .ImageCaptioner import BG_MODE_PROMPTS
import os
from pathlib import Path


# A logger for this file
log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# JSON schema for structured outputs
# -------------------------------------------------------------------
AFFORDANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "affordance_tags": {
            "type": "array",
            "description": "List of affordance tags.",
            "items": {"type": "string"},
        },
        "affordance_types": {
            "type": "array",
            "description": "List of affordance types corresponding to each tag.",
            "items": {
                "type": "string",
                "enum": [],
            },
        },
        "affordance_count": {
            "type": "array",
            "description": "List of number of affordances detected corresponding to each tag.",
            "items": {"type": "integer"},
        },
    },
    "required": ["affordance_tags", "affordance_types", "affordance_count"],
}


class AffordancePredictor:
    def __init__(
        self,
        system_prompt: str,
        user_query: str,
        affordance_types: List[str],
        bg_mask_mode: str = "no_mask",
        max_images: int = 1,
        debug: bool = False,
        debug_dir: str = ".",
    ):
        self.system_prompt = system_prompt
        self.user_query = user_query
        self.max_images = (
            max_images  # Subsample object views if too many images are provided
        )
        self.bg_mask_mode = bg_mask_mode

        if bg_mask_mode not in BG_MODE_PROMPTS:
            raise ValueError(
                f"Invalid bg_mask_mode. Please select value in {BG_MODE_PROMPTS.keys()}."
            )

        self.full_system_prompt = self.system_prompt

        if self.bg_mask_mode != "no_mask":
            self.full_system_prompt += " " + BG_MODE_PROMPTS[self.bg_mask_mode]

        self.affordance_schema = AFFORDANCE_SCHEMA.copy()
        self.affordance_schema["properties"]["affordance_types"]["items"][
            "enum"
        ] = affordance_types

        # Debug stuff
        self.debug = debug
        self.debug_dir = Path(debug_dir) / "affordance_predictor"
        self.debug_counter = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            log.info(
                f"Debugging enabled. Saving prompts and outputs to {self.debug_dir}"
            )

    def preprocess_image(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = mask == 2
        if self.bg_mask_mode == "no_mask":
            result = np.copy(img)
        elif self.bg_mask_mode == "white":
            result = np.where(mask[:, :, np.newaxis], img, 255)
        elif self.bg_mask_mode == "black":
            result = np.where(mask[:, :, np.newaxis], img, 0)
        elif self.bg_mask_mode == "greyscale":
            grey_img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
            result = np.where(mask[:, :, np.newaxis], img, grey_img)

        return result

    def postprocess_response(self, response: str) -> dict:
        """
        Parse the returned JSON string. Fallback to empty structure if invalid.
        """
        try:
            import json

            return json.loads(response)
        except Exception:
            raise Exception(f"Failed to parse JSON response: {response}")
            # log.error(f"Failed to parse JSON response: {response}")
            # return {"affordance_tags": [], "affordance_types": []}

    def annotate_map(self, map: "ObjectMap") -> None:
        pbar = tqdm(map)
        pbar.set_description("Predicting affordances")
        for obj in pbar:
            segments = obj.segments.get_sorted()
            views = [s.rgb for s in segments]
            masks = [s.mask for s in segments]
            imgs = [
                self.preprocess_image(view, mask) for (view, mask) in zip(views, masks)
            ]
            obj_tag = obj.tag
            affordance_info = self(imgs, object_tag=obj_tag)
            obj.affordance_tags = affordance_info["affordance_tags"]
            obj.affordance_types = affordance_info["affordance_types"]

            # log.info(
            #     f"For object {obj_tag}, the affordance tags are {obj.affordance_tags}"
            # )
            # log.info(
            #     f"For object {obj_tag}, the affordance types are {obj.affordance_types}"
            # )

    def annotate_map_with_descs(
        self, map: "ObjectMap", descs: List[Tuple[str, str]]
    ) -> None:
        pbar = tqdm(map)
        pbar.set_description("Predicting affordances")

        desc_id_to_str = {desc_id: desc_str for desc_id, desc_str in descs}

        for obj in pbar:
            segments = obj.segments.get_sorted()
            views = [s.rgb for s in segments]
            masks = [s.mask for s in segments]
            imgs = [
                self.preprocess_image(view, mask) for (view, mask) in zip(views, masks)
            ]

            obj_tag = obj.tag
            obj_desc_ids = obj.desc_ids

            if len(obj_desc_ids) == 0:
                continue

            for obj_desc_id in obj_desc_ids:

                query = desc_id_to_str.get(obj_desc_id)
                self.user_query = f" The user asks you to '{query}'. Think step-by-step about the affordances of the object that are relevant to the user query."

                # NOTE: Logs, can be removed
                log.info(
                    f"Predicting affordances for object {obj_tag} and desc '{query}'"
                )

                affordance_info = self(imgs, object_tag=obj_tag)

                obj.affordance_tags = affordance_info["affordance_tags"]
                obj.affordance_types = affordance_info["affordance_types"]

                affordance_info_with_desc_id = (
                    obj_desc_id,
                    affordance_info["affordance_tags"],
                    affordance_info["affordance_types"],
                )
                obj.affordance_info_with_desc_ids.append(affordance_info_with_desc_id)

            # log.info(
            #     f"For object {obj_tag}, the affordance tags are {obj.affordance_tags}"
            # )
            # log.info(
            #     f"For object {obj_tag}, the affordance types are {obj.affordance_types}"
            # )

    def __call__(self, imgs: List[np.ndarray]) -> str:
        raise NotImplementedError
