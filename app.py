import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# 加载模型
@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to("cuda")
    pipe.safety_checker = None
    return pipe

pipe = load_pipeline()

def text_to_img(prompt):
    guidance_scale = 11
    images = pipe(prompt=prompt, guidance_scale=guidance_scale).images
    return images[0]

# Streamlit App 前端
st.title("Text to Image Generator")
prompt = st.text_input("Enter a prompt", "A photorealistic image of a futuristic city skyline at sunset.")

if st.button("Generate Image"):
    image = text_to_img(prompt)
    st.image(image, caption="Generated Image", use_column_width=True)

