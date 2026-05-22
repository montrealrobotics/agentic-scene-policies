import os
import json
import cv2
import numpy as np
from google import genai
from google.genai import types
from typing import List, Tuple
import logging
import torch

from concept_graphs.mapping.ObjectMap import ObjectMap
from concept_graphs.perception.ft_extraction import FeatureExtractor

log = logging.getLogger(__name__)


class TaskObjPredictor:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        thinking_budget: int,
        ft_extractor: FeatureExtractor,
        map_obj_ft: str = "visual",
        top_k: int = 1,
    ):
        self.full_system_prompt = system_prompt
        self.model = model
        self.ft_extractor = ft_extractor
        self.top_k = top_k
        self.map_obj_ft = map_obj_ft
        # expects GEMINI_API_KEY in your env
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.thinking_budget = thinking_budget

    def __call__(self, task_desc: str) -> dict:

        # prepare the text prompt
        prompt = f"The task description is as follows: {task_desc}. Please give the relevant object names."

        if self.model == "gemini-1.5-pro":
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
            )
        else:
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
            )

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=cfg,
                )
                output = getattr(response, "text", "")
                # NOTE: Logs can be removed
                log.info(f"Prompt: {prompt}")
                log.info(f"Gemini response: {output}")
                break
            except Exception as e:
                log.error(
                    f"Attempt {attempt + 1} for GeminiAffordancePredictor failed with error: {e}"
                )
        else:
            log.error("All attempts failed, continuing with empty response...")
            output = ""

        return output

    def get_descs_objs(
        self, descs: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Get object descriptions for each description in descs.
        """

        descs_with_objs = []
        for d_id, d_text in descs:
            # Find objects in the map that are relevant to this description
            relevant_obj = self(d_text)
            descs_with_objs.append((d_id, d_text, relevant_obj))
        return descs_with_objs

    def get_map_obj_ft(self, map: ObjectMap) -> torch.Tensor:
        """
        Get the object features from the map based on the specified feature type.
        """

        if self.map_obj_ft == "visual":
            return map.semantic_features
        elif self.map_obj_ft == "text":
            tags = []
            for obj in map:
                tag = getattr(obj, "tag", "empty")
                if tag == "empty":
                    log.warning(
                        f"Object {getattr(obj, 'id', 'unknown')} has empty tag; using placeholder for text features."
                    )
                    tag = "object"
                tags.append(tag)
            if not tags:
                return None
            with torch.no_grad():
                ft = self.ft_extractor.encode_text(tags)  # (O, K)
            return ft.to(map.device)
        else:
            raise ValueError(f"Unknown map_obj_ft: {self.map_obj_ft}")

    def annotate_map_with_descs(
        self, map: ObjectMap, descs: List[Tuple[str, str]]
    ) -> None:
        """
        For every (desc_id, desc_str) in descs, first find the relevant_obj using get_desc_objs.
        Then find the top-k most similar objects in the map
        (based on cosine similarity between relevant_obj text features and each object's semantic_ft)
        and append desc_id to obj.desc_ids (a list) for those objects. Objects may collect multiple
        desc_ids; some objects may receive none. Every description must map to up to k objects
        (fewer if map has < k objects).
        """

        if not descs or len(map) == 0:
            log.warning("No descriptions or empty map, skipping annotation.")
            return

        device = map.device

        descs_with_objs = self.get_descs_objs(descs)

        # Ensure semantic tensor & key map are current
        if map.semantic_tensor is None or len(map.key_map) != len(map.objects):
            map.collate()

        obj_ft = map.semantic_tensor  # (O, K)
        if obj_ft is None or obj_ft.numel() == 0:
            log.warning("Map has no semantic features, skipping annotation.")
            return

        # Encode descriptions -> (D, K)
        desc_ids, desc_texts, relevant_objs = zip(*descs_with_objs)
        with torch.no_grad():
            text_ft = self.ft_extractor.encode_text(list(relevant_objs))  # (D, K)

        # Move to same device
        obj_ft = obj_ft.to(device)
        text_ft = text_ft.to(device)

        # Normalize for cosine similarity
        obj_ft_norm = torch.nn.functional.normalize(obj_ft, dim=1)
        text_ft_norm = torch.nn.functional.normalize(text_ft, dim=1)

        # Similarity matrix (D, O)
        sim = text_ft_norm @ obj_ft_norm.T

        num_objects = obj_ft.shape[0]
        k = min(self.top_k, num_objects)
        if k <= 0:
            log.warning("Top-k is non-positive, skipping annotation.")
            return

        # For each description, get top-k object indices
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)

        # Assign desc_ids to objects
        for d_row, d_id in enumerate(desc_ids):
            for obj_idx in topk_idx[d_row].tolist():
                obj = map[obj_idx]  # uses key_map indirection
                # Initialize desc_ids list if absent
                if not hasattr(obj, "desc_ids") or obj.desc_ids is None:
                    obj.desc_ids = []
                # Avoid duplicate addition
                if d_id not in obj.desc_ids:
                    obj.desc_ids.append(d_id)

        return
