import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_labsim_example() -> dict:
    """Creates a random input example for the LabSim policy."""
    return {
        "state": np.random.rand(8),  # 8-DOF robot state
        "camera_0_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),  # Primary camera
        "camera_1_rgb": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),  # Secondary camera
        "task": "Perform laboratory task",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LabSimInputs(transforms.DataTransformFn):

    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        state = transforms.pad_to_dim(data["state"], self.action_dim)

        camera_1_rgb = _parse_image(data["camera_1_rgb"])
        camera_2_rgb = _parse_image(data["camera_2_rgb"])

        inputs = {
            "state": state,
            "image": {
                "camera_1_rgb": camera_1_rgb,
                "camera_2_rgb": camera_2_rgb,
            },
            "image_mask": {
                "camera_1_rgb": np.True_,
                "camera_2_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LabSimOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])} 