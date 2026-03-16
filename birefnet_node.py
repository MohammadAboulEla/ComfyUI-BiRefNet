import torch, os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch.nn.functional as F
from PIL import Image
from models.birefnet import BiRefNet
from config import Config
from torchvision.transforms.functional import normalize
import numpy as np
import folder_paths

config = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"
folder_paths.folder_names_and_paths["BiRefNet"] = (
    [os.path.join(folder_paths.models_dir, "BiRefNet")],
    folder_paths.supported_pt_extensions,
)


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def resize_image(image):
    image = image.convert("RGB")
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def check_download_model(model_path, repo_id="ZhengPeng7/BiRefNet"):
    if not os.path.exists(model_path):
        folder_path = os.path.dirname(model_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.basename(model_path)
        print(f"Start Download BiRefNet model to: {model_path}")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"*{file_name}*"],
            local_dir=folder_path,
            local_dir_use_symlinks=False,
        )
        return True
    return False


class BiRefNet_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep the model in VRAM after inference. Disable to free memory (Recommended).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "BiRefNet"

    def remove_background(self, image, keep_model_loaded=False):
        model_name = "model.safetensors"
        model_path = os.path.join(folder_paths.models_dir, "BiRefNet", model_name)
        check_download_model(model_path, repo_id="ZhengPeng7/BiRefNet")

        birefnetmodel = BiRefNet(bb_pretrained=False)
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path, map_location=device)
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        birefnetmodel.load_state_dict(state_dict)
        birefnetmodel.to(device)
        birefnetmodel.eval()

        processed_images = []
        processed_masks = []

        for image in image:
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = torch.unsqueeze(im_tensor, 0)
            im_tensor = torch.divide(im_tensor, 255.0)
            im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if torch.cuda.is_available():
                im_tensor = im_tensor.cuda()

            result = birefnetmodel(im_tensor)[-1].sigmoid()

            result = torch.squeeze(
                F.interpolate(result, size=(h, w), mode="bilinear"), 0
            )
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result - mi) / (ma - mi)
            im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            new_im = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_im)

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        if not keep_model_loaded:
            del birefnetmodel
            torch.cuda.empty_cache()

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "BiRefNet_Node": BiRefNet_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_Node": "BG Removal",
}
