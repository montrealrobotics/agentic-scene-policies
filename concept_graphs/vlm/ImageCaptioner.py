from typing import List
import numpy as np
import logging
from tqdm import tqdm


# A logger for this file
log = logging.getLogger(__name__)


BG_MODE_PROMPTS = {
    "no_mask": "",
    "white": "The background and foreground objects have been removed and replaced with white pixels. Ignore this and focus on the central object.",
    "black": "The background and foreground objects have been removed and replaced with black pixels. Ignore this and focus on the central object.",
    "greyscale": "The background and foreground objects are greyscale. Focus on the colored central object.",
}

FIRST_IMG_BG_MODE_PROMPTS_SUFFIX = "So, only the central object is visible in the first image. Focus only on this object. The other images of this object are not masked, and the full context is shown to help you."
FIRST_IMG_BG_MODE_PROMPTS = {
    "no_mask": "",
    "white": f"For the first image, the background and foreground objects have been removed and replaced with white pixels. {FIRST_IMG_BG_MODE_PROMPTS_SUFFIX}",
    "black": f"For the first image, the background and foreground objects have been removed and replaced with black pixels. {FIRST_IMG_BG_MODE_PROMPTS_SUFFIX}",
    "greyscale": f"For the first image, the background and foreground objects are greyscale. {FIRST_IMG_BG_MODE_PROMPTS_SUFFIX}",
}

NO_OBJECT_MSG = "If you believe there is no object in the images or the images depict a background element, output BACKGROUND."
RESPONSE_PREFIX = "The object is"


class ImageCaptioner:
    def __init__(
        self,
        system_prompt: str,
        user_query: str,
        max_images: int,
        tag: bool,
        bg_mask_mode: str,
        first_img_bg_mask_mode: str,
        allow_no_object_pred: bool,
        no_object_prompt: str = NO_OBJECT_MSG,
        force_reponse_prefix: bool = True,
        prefix: str = RESPONSE_PREFIX,
    ):
        self.system_prompt = system_prompt
        self.user_query = user_query
        self.max_images = (
            max_images  # Subsample object views if too many images are provided
        )
        self.tag = tag  # Use the tag field of objects instead of the caption field when captioning a map
        self.bg_mask_mode = bg_mask_mode
        self.first_img_bg_mask_mode = first_img_bg_mask_mode
        self.allow_no_object_pred = allow_no_object_pred
        self.no_object_prompt = no_object_prompt
        self.force_response_prefix = force_reponse_prefix
        self.prefix = prefix

        if bg_mask_mode not in BG_MODE_PROMPTS:
            raise ValueError(
                f"Invalid bg_mask_mode. Please select value in {BG_MODE_PROMPTS.keys()}."
            )

        if first_img_bg_mask_mode not in FIRST_IMG_BG_MODE_PROMPTS:
            raise ValueError(
                f"Invalid first_img_bg_mask_mode. Please select value in {FIRST_IMG_BG_MODE_PROMPTS.keys()}."
            )

        self.full_system_prompt = self.system_prompt

        if self.allow_no_object_pred:
            self.full_system_prompt += " " + self.no_object_prompt
        if self.bg_mask_mode != "no_mask":
            self.full_system_prompt += " " + BG_MODE_PROMPTS[self.bg_mask_mode]
        if self.first_img_bg_mask_mode != "no_mask":
            self.full_system_prompt += (
                " " + FIRST_IMG_BG_MODE_PROMPTS[self.first_img_bg_mask_mode]
            )
        if self.force_response_prefix:
            self.full_system_prompt += (
                " " + f"Begin your response with '{self.prefix}'."
            )

        if self.tag:
            log.info(
                f"Instantiated Image Tagger with full system prompt: {self.full_system_prompt}"
            )
        else:
            log.info(
                f"Instantiated Image Captioner with full system prompt: {self.full_system_prompt}"
            )

    def preprocess_image(
        self, img: np.ndarray, mask: np.ndarray, first_img_flag: bool = False
    ) -> np.ndarray:
        current_bg_mask_mode = (
            self.first_img_bg_mask_mode if first_img_flag else self.bg_mask_mode
        )

        if current_bg_mask_mode == "no_mask":
            result = np.copy(img)
        elif current_bg_mask_mode == "white":
            result = np.where(mask[:, :, np.newaxis], img, 255)
        elif current_bg_mask_mode == "black":
            result = np.where(mask[:, :, np.newaxis], img, 0)
        elif current_bg_mask_mode == "greyscale":
            grey_img = np.mean(img, axis=2, keepdims=True).astype(np.uint8)
            result = np.where(mask[:, :, np.newaxis], img, grey_img)

        return result

    def postprocess_response(self, response: str) -> str:
        response = response.replace(self.prefix + " ", "")
        response = response.replace(self.prefix, "")
        response = response.replace("*", "")
        response = response.replace("'", "")
        response = response.replace('"', "")
        response = response.replace(".", "")
        response = response.replace(",", "")

        if response.startswith("a "):
            response = response[2:]
        if response.startswith("an "):
            response = response[3:]

        return response

    def caption_map(self, map: "ObjectMap") -> None:
        pbar = tqdm(map)
        pbar.set_description("Tagging Objects" if self.tag else "Captioning Objects")
        for obj in pbar:
            segments = obj.segments.get_sorted()
            views = [s.rgb for s in segments]
            masks = [s.mask for s in segments]

            imgs = [
                self.preprocess_image(view, mask) for (view, mask) in zip(views, masks)
            ]

            if self.bg_mask_mode != "no_mask":
                first_img = self.preprocess_image(
                    views[0], masks[0], first_img_flag=True
                )
                imgs = [first_img] + imgs

            if self.tag:
                obj.tag = self(imgs)
                log.info(obj.tag)
            else:
                obj.caption = self(imgs)
                log.info(obj.caption)

    def __call__(self, List: [np.ndarray]) -> str:
        raise NotImplementedError
