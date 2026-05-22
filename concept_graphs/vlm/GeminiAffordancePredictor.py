import os
import json
import cv2
import numpy as np
from google import genai
from google.genai import types
from typing import List
from .AffordancePredictor import AffordancePredictor
import logging

log = logging.getLogger(__name__)


class GeminiAffordancePredictor(AffordancePredictor):
    def __init__(
        self,
        model: str,
        thinking_budget: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        # expects GEMINI_API_KEY in your env
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.thinking_budget = thinking_budget
        self.affordance_schema["propertyOrdering"] = [
            "affordance_tags",
            "affordance_types",
            "affordance_count",
        ]
        log.info(f"Using schema: {self.affordance_schema}")

    def encode_images(self, images: List[np.ndarray]) -> List[bytes]:
        # convert BGR -> RGB, JPEG-encode, return bytes
        out: List[bytes] = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ok, buf = cv2.imencode(".jpg", rgb)
            if not ok:
                log.warning("Failed to JPEG-encode an image")
                continue
            out.append(buf.tobytes())
        return out

    def __call__(self, images: List[np.ndarray], object_tag: str) -> dict:
        # cap number of images
        imgs = images[: self.max_images]

        # prepare the text prompt
        prompt = f"The object you should focus on is '{object_tag}'. {self.user_query}"

        # build the contents list: first text, then each image
        parts = [
            types.Part.from_text(text=prompt),
            *[
                types.Part.from_bytes(data=b, mime_type="image/jpeg")
                for b in self.encode_images(imgs)
            ],
        ]

        if self.model == "gemini-1.5-pro":
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
                response_mime_type="application/json",
                response_schema=self.affordance_schema,
            )
        else:
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
                response_mime_type="application/json",
                response_schema=self.affordance_schema,
            )

        for attempt in range(10):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=cfg,
                )
                output = getattr(
                    response, "text", '{"affordance_tags": [], "affordance_types": []}'
                )
                response_post = self.postprocess_response(output)
                break
            except Exception as e:
                log.error(
                    f"Attempt {attempt + 1} for GeminiAffordancePredictor failed with error: {e}"
                )
        else:
            log.error("All attempts failed, continuing with empty response...")
            output = '{"affordance_tags": [], "affordance_types": []}'
            response_post = self.postprocess_response(output)

        if self.debug:
            # Dump prompt and output to debug directory
            debug_file = os.path.join(
                self.debug_dir, f"gemini_affordance_{self.debug_counter}.json"
            )
            with open(debug_file, "w") as f:
                json.dump(
                    {
                        "system_prompt": self.system_prompt,
                        "prompt": prompt,
                        "output": output,
                    },
                    f,
                    indent=2,
                )
            log.info(f"Debug output saved to {debug_file}")
            self.debug_counter += 1

        return response_post
