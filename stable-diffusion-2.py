from diffusers import AutoPipelineForText2Image
import torch
from pathlib import Path
from PIL import Image

def get_token_count(prompt):
    tokens = prompt.split()
    return len(tokens)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-2", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

def get_valid_prompt():
    while True:
        prompt = input("Please enter your prompt (maximum 70 tokens): ")
        token_count = get_token_count(prompt)
        
        if token_count > 70:
            print(f"Your prompt has {token_count} tokens, which exceeds the 70-token limit.")
            print("Please shorten your prompt and try again.")
        else:
            return prompt

def get_valid_number_of_images():
    while True:
        try:
            num_images = int(input("How many images would you like to generate? "))
            if num_images > 0:
                return num_images
            else:
                print("Please enter a valid number greater than 0.")
        except ValueError:
            print("Please enter a valid integer.")

prompt = get_valid_prompt()

num_images = get_valid_number_of_images()

generator = torch.Generator("cuda").manual_seed(31)

output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)

DIR_NAME = "./generated_images/"
dirpath = Path(DIR_NAME)
dirpath.mkdir(parents=True, exist_ok=True)

for idx, image in enumerate(output.images):
    image_name = f"generated_image_{idx+1}.png"
    image_path = dirpath / image_name
    image.save(image_path)

print(f"Images have been saved to {DIR_NAME}")
