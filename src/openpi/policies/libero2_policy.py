import dataclasses

import einops
import numpy as np
import cv2
from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "state": np.random.rand(8),
        "agentview_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    # Resize image to 224x224
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def invert_gripper_actions(gripper_action):
    """Invert gripper actions (equivalent to tf function but using numpy)."""
    return 1.0 - gripper_action


@dataclasses.dataclass(frozen=True)
class Libero2Inputs(transforms.DataTransformFn):
    
    action_dim: int
    
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        
        base_image = _parse_image(data["agentview_rgb"])
        wrist_image = _parse_image(data["eye_in_hand_rgb"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "action" in data:
            action = data["action"]
            # Handle 9-dimensional actions: remove second-to-last dim and process last dim
            if action.shape[-1] == 9:
                # Remove the second-to-last dimension (index 7)
                action = np.concatenate([action[..., :7], action[..., 8:]], axis=-1)
                # Process the gripper action (now the last dimension)
                # gripper_action = action_without_second_last[..., -1:]
                # gripper_action = np.clip(gripper_action, 0, 1)
                # gripper_action = invert_gripper_actions(gripper_action)
                # action = np.concatenate([action_without_second_last[..., :-1], gripper_action], axis=-1)
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["action"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Libero2Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"action": np.asarray(data["action"][:, :9])}
