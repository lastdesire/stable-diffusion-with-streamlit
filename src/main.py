from typing import Optional
import streamlit as st
import torch
from diffusers import (
    StableDiffusionPipeline,
)
from PIL import Image

DEFAULT_PROMPT = "the fly sat on the jam, that's the whole poem"
OUTPUT_IMG = "output"


@st.cache(allow_output_mutation=True, max_entries=1)
def get_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def generate(prompt, default_height, default_width, num_inference_steps, guidance_scale, number_of_pictures):
    pipe = get_pipeline()
    image = pipe(prompt=[prompt] * number_of_pictures, height=default_height, width=default_width,
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps).images
    set_image(OUTPUT_IMG, image.copy())
    return image


def prompt_and_generate_button(prefix, default_height, default_width, num_inference_steps, guidance_scale,
                               number_of_pictures):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating your image, please wait..."):
            image = generate(prompt, default_height, default_width, num_inference_steps, guidance_scale,
                             number_of_pictures)
        st.image(image)


def txt2img_tab(default_height, default_width, num_inference_steps, guidance_scale, number_of_pictures):
    prompt_and_generate_button("txt2img", default_height, default_width, num_inference_steps, guidance_scale,
                               number_of_pictures)


def settings_tab():
    default_height = st.slider("Final height", value=512, min_value=16, max_value=1024,
                               help="After getting an image there will be resizing for specified height.")
    default_width = st.slider("Final width", value=512, min_value=16, max_value=1024,
                              help="After getting an image there will be resizing for specified width.")
    num_inference_steps = st.slider("Num inference steps", value=50, min_value=1, max_value=100,
                                    help="More steps usually lead to a higher quality of image and slower inference.")
    guidance_scale = st.slider("Guidance scale", value=7, min_value=0, max_value=30,
                               help="A higher hint scale encourages images that are closely related to the text hint, "
                                    "usually at the expense of lower image quality.")
    number_of_pictures = st.slider("Num of pictures", value=1, min_value=1, max_value=10,
                                   help="Number of images to be generated")
    return default_height, default_width, num_inference_steps, guidance_scale, number_of_pictures


def main():
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion + Streamlit SPbU 3 Semester homework by [lastdesire](https://github.com/lastdesire)")
    tab, settings = st.tabs(
        ["Text to Image", "Settings"]
    )
    with settings:
        default_height, default_width, num_inference_steps, guidance_scale, number_of_pictures = settings_tab()

    with tab:
        txt2img_tab(default_height, default_width, num_inference_steps, guidance_scale, number_of_pictures)

    with st.sidebar:
        st.header("Latest image:")
        output_image = get_image(OUTPUT_IMG)
        if output_image:
            st.image(output_image)
        else:
            st.markdown("No output created yet.")
    st.write("NOTE: if you see a completely black image after generation, try increasing the \"Num inference steps\" "
             "in the settings (NSFW content blocking)")


if __name__ == "__main__":
    main()
