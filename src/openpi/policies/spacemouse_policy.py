import dataclasses

import cv2
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_spacemouse_example() -> dict:
    """Creates a random input example for the SpaceMouse policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the organe pumpkin",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format, center crop and resize to 224x224."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    
    # 居中裁切成正方形 (480x480)
    h, w = image.shape[:2]
    crop_size = min(h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    image = image[start_h:start_h + crop_size, start_w:start_w + crop_size]
    
    # 缩放到 224x224
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    return image


@dataclasses.dataclass(frozen=True)
class SpaceMouseInputs(transforms.DataTransformFn):
    """
    Transform for SpaceMouse dataset inputs.
    Converts dataset format to the format expected by pi0 models.
    """

    # Determines which model will be used
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images from LeRobot format (float32 C,H,W) to uint8 (H,W,C)
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict with standard pi0 keys
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad non-existent right wrist image with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask padding images for pi0, but not for pi0-FAST
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Add actions during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Add prompt/language instruction
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SpaceMouseOutputs(transforms.DataTransformFn):
    """
    Transform for SpaceMouse dataset outputs.
    Converts model outputs back to dataset-specific format.
    """

    def __call__(self, data: dict) -> dict:
        # Return only the first 7 actions (SpaceMouse action dimension)
        return {"actions": np.asarray(data["actions"][:, :7])}

