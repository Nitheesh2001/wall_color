#!/usr/bin/env python3
import argparse, pathlib, torch
from PIL import Image
from mask_utils import get_wall_mask
from inpaint_utils import load_inpaint_pipeline, run_inpaint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--prompt", required=True, help="Color, e.g. 'emerald green matte'")
    p.add_argument("--output", default="outputs/result.png", help="Output path")
    return p.parse_args()

def main():
    args = parse_args()
    img = Image.open(args.image).convert("RGB")
    mask = get_wall_mask(img)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_inpaint_pipeline(device=device)

    full_prompt = f"Repaint only the wall in {args.prompt}; leave everything else unchanged."
    result = run_inpaint(pipe, img, mask, full_prompt)

    pathlib.Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    result.save(args.output)
    print("✅ Saved →", args.output)

if __name__ == "__main__":
    main()
