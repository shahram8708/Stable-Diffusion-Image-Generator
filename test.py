import torch
from torch.amp import autocast 
from diffusers import StableDiffusionPipeline
from PIL import Image

def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    return pipe, device

def generate_image(prompt, pipe, device):
    if device == "cuda":
        with autocast("cuda"): 
            image = pipe(prompt).images[0]
    else:
        image = pipe(prompt).images[0]
    return image

if __name__ == "__main__":
    pipe, device = load_model()

    prompt = input("Enter the prompt for image generation: ")

    try:
        generated_image = generate_image(prompt, pipe, device)

        generated_image.save("generated_image.png")
        generated_image.show()

        print("Image generated and saved as 'generated_image.png'.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
