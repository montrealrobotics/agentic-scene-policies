import time
import os
import json
import cv2
import numpy as np
from google import genai
from google.genai import types
from typing import List
import logging

log = logging.getLogger(__name__)

SCHEMA = {"type": "array", "items": {"type": "boolean"}}


class GeminiClassifier:
    def __init__(
        self,
        model: str,
        thinking_budget: int = 1024,
        system_prompt: str = "A user is searching for objects in a map. You are presented with images of the best object candidates given the user query. Focus on the central object in each image. Confirm whether each object is relevant given the user query. If an object is relevant, return true, otherwise return false. You should return true or false for each object. Multiple objects may be relevant. It's also possible that no objects are relevant.",
    ):
        self.model = model
        # expects GEMINI_API_KEY in your env
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.thinking_budget = thinking_budget
        self.full_system_prompt = system_prompt
        self.schema = SCHEMA
        log.info(f"Using schema: {self.schema}")

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

    def __call__(self, images: List[np.ndarray], user_query: str) -> List[bool] | None:
        # prepare the text prompt
        user_query = f"The user query is: {user_query}."

        # build the contents list: first text, then each image
        parts = [
            types.Part.from_text(text=user_query),
            *[
                types.Part.from_bytes(data=b, mime_type="image/jpeg")
                for b in self.encode_images(images)
            ],
        ]

        if self.model == "gemini-1.5-pro":
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
                response_mime_type="application/json",
                response_schema=self.schema,
            )
        else:
            cfg = types.GenerateContentConfig(
                system_instruction=self.full_system_prompt,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
                response_mime_type="application/json",
                response_schema=self.schema,
            )

        for attempt in range(10):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=cfg,
                )
                output = getattr(response, "text", "[]")
                output = json.loads(output)

                assert isinstance(output, list), "Response should be a list"
                assert all(
                    isinstance(item, bool) for item in output
                ), "All items should be boolean"
                assert len(output) == len(
                    images
                ), "Output length should match number of images"

                break

            except Exception as e:
                log.error(
                    f"Attempt {attempt + 1} for GeminiClassifier failed with error: {e}"
                )
                time.sleep(0.1 * 2**attempt)
        else:
            log.error("All attempts failed. Returning None...")
            output = None

        return output


if __name__ == "__main__":
    # Example usage
    classifier = GeminiClassifier(model="gemini-1.5-pro")
    base_path = "/home/fordpinto/Downloads/imgs"

    # Load all images in base_path as ndarrays. Order should be alphabetical.
    images = []
    for filename in sorted(os.listdir(base_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(base_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)

    for query in [
        "Grab all the fruits.",
        "Grab all the vegetables.",
        "Find the cat.",
        "Give me the red thing.",
        "I need something orange and green.",
    ]:

        results = classifier(images, query)

        print(f"Query: {query}")
        print("Results:", results)
        print()
