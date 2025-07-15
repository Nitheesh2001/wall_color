import os, torch
from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel, AutoencoderKL
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def load_inpaint_pipeline(device="cpu"):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=torch.float32,
        use_auth_token=hf_token
    )
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float32,
        use_auth_token=hf_token,
        vae=AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32, use_auth_token=hf_token)
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe

def run_inpaint(pipe, pil_img: Image.Image, mask: Image.Image, prompt: str) -> Image.Image:
    """
    Runs the inpainting pipeline.
    """
    result_img = pipe(
        prompt=prompt,
        image=pil_img.resize((1024, 1024)),
        mask_image=mask.resize((1024, 1024)),
        control_image=mask.resize((1024, 1024)),
        num_inference_steps=30,
    ).images[0]

    return result_img
