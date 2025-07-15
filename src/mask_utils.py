# src/mask_utils.py
import torch, numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

# Path to downloaded SAM checkpoint
_SAM_CKPT = "models/sam/sam_vit_b_01ec64.pth"

def get_wall_mask(pil_img: Image.Image) -> Image.Image:
    """
    Uses SAM to auto-generate masks and selects the one with
    largest area (assumed to be the wall).
    Returns a binary PIL mask (mode 'L').
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=_SAM_CKPT).to(device)
    mask_gen = SamAutomaticMaskGenerator(sam)

    arr = np.array(pil_img)
    masks = mask_gen.generate(arr)

    # Filter masks to find the most likely wall
    wall_masks = []
    for mask_info in masks:
        mask = mask_info["segmentation"]
        # Simple heuristic: check if the mask is large and somewhat rectangular
        # You might need to adjust these thresholds based on your image dataset
        if mask_info["area"] > (pil_img.width * pil_img.height * 0.2):  # At least 20% of image area
            # Calculate aspect ratio of bounding box
            x, y, w, h = mask_info["bbox"]
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 3.0:  # Not too thin or tall
                wall_masks.append(mask_info)

    if not wall_masks:
        # Fallback to largest mask if no wall-like masks are found
        best = max(masks, key=lambda m: m["area"])
    else:
        # Choose the largest among the wall-like masks
        best = max(wall_masks, key=lambda m: m["area"])

    seg = (best["segmentation"] * 255).astype(np.uint8)
    return Image.fromarray(seg)
