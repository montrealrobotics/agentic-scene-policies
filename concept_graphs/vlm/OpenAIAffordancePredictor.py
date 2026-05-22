import cv2
import base64
import openai
import numpy as np
from .AffordancePredictor import AffordancePredictor
import logging

log = logging.getLogger(__name__)


class OpenAIAffordancePredictor(AffordancePredictor):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = openai.OpenAI()
        self.affordance_schema["additionalProperties"] = False

    def encode_images(self, images: [np.ndarray]) -> [str]:
        # To RGB
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]

    def __call__(self, images: [np.ndarray], object_tag: str) -> str:
        if len(images) > self.max_images:
            images = images[: self.max_images]

        base64_images = self.encode_images(images)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": self.full_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"The object you should focus on is {object_tag}. {self.user_query}",
                        },
                        *[
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_img}",
                            }
                            for b64_img in base64_images
                        ],
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "affordance_info",
                    "strict": True,
                    "schema": self.affordance_schema,
                }
            },
        )

        # Parse content
        try:
            output = response.output_text
        except AttributeError:
            output = "{}"

        return self.postprocess_response(output)
