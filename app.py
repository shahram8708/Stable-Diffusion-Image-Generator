import torch
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import streamlit as st
import base64

@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)
    
    return pipe, device

def generate_image(prompt, pipe, device):
    with st.spinner("Generating the image..."):
        if device == "cuda":
            with autocast(device_type="cuda"):
                image = pipe(prompt).images[0]
        else:
            image = pipe(prompt).images[0]
    return image

def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            color: #000;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

def main():
    add_custom_css()

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    pipe, device = load_model()

    st.title("🎨 Stable Diffusion Image Generator")
    st.write("Create stunning images from your imagination. Simply provide a prompt and watch the magic happen.")

    prompt = st.text_input("Enter a creative prompt", value="A futuristic city at sunset", max_chars=150)

    if st.button("Generate Image"):
        if prompt:
            generated_image = generate_image(prompt, pipe, device)

            st.image(generated_image, caption="Generated Image", use_column_width=True)

            image_path = "generated_image.png"
            generated_image.save(image_path)

            with open(image_path, "rb") as file:
                image_bytes = file.read()
                b64_image = base64.b64encode(image_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64_image}" download="generated_image.png">📥 Download Image</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.error("Please enter a valid prompt to generate an image.")

if __name__ == "__main__": 
    main()
