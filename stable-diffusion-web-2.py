import gradio as gr
from diffusers import AutoPipelineForText2Image
import torch
from pathlib import Path
from PIL import Image

def setup_pipeline():
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-2", 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to("cuda")
    return pipeline

def generate_images(prompt, num_images):
    pipeline = setup_pipeline()
    
    generator = torch.Generator("cuda").manual_seed(31)

    output = pipeline(prompt, num_images_per_prompt=num_images, generator=generator)

    DIR_NAME = "./generated_images/"
    dirpath = Path(DIR_NAME)
    dirpath.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for idx, image in enumerate(output.images):
        image_name = f"generated_image_{idx+1}.png"
        image_path = dirpath / image_name
        image.save(image_path)
        image_paths.append(str(image_path))

    return image_paths

def display_images(image_paths):
    images = [Image.open(path) for path in image_paths]
    return images

def gradio_interface(prompt, num_images):
    if len(prompt.split()) > 70:
        return "Error: Prompt exceeds 70 tokens. Please shorten your prompt."

    try:
        image_paths = generate_images(prompt, num_images)
        images = display_images(image_paths)
        return images
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #4CAF50; font-weight: bold;">🎨 Next-Gen Image Generation</h1>
            <p style="font-size: 1.2em; color: #555;">
                Bring your imagination to life with this professional-grade image generator powered by Stable Diffusion 2. Let your creativity shine!
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("""
                ### Enter Prompt
                Describe the image you want to create:
            """)
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe your dream image (max 70 tokens)",
                lines=3,
            )
            gr.Markdown("""
                ### Select Number of Images
                Specify how many images you'd like to generate:
            """)
            num_images_input = gr.Slider(
                minimum=1, step=1, value=1, label="Number of Images"
            )
            generate_button = gr.Button(
                "Generate Images", 
                elem_id="generate-btn"
            )

        with gr.Column(scale=2):
            gr.Markdown("""
                ### Generated Images
                Your creations will appear below:
            """)
            image_gallery = gr.Gallery(
                label="Generated Images",
                elem_id="image-gallery"
            )

    generate_button.click(
        fn=gradio_interface,
        inputs=[prompt_input, num_images_input],
        outputs=[image_gallery],
    )

demo.launch(debug=True)
