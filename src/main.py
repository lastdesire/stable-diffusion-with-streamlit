from typing import Optional
import streamlit as st
from diffusers import (
    StableDiffusionPipeline,
)
from PIL import Image

DEFAULT_PROMPT = "the fly sat on the jam, that's the whole poem"
OUTPUT_IMG = "output"

global GUIDANCE_SCALE
global DEFAULT_WIDTH
global DEFAULT_HEIGHT
global NUM_INFERENCE_STEPS


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def generate(prompt):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # disable the following line if you run on CPU
    # pipe = pipe.to("cuda")
    image = pipe(prompt=prompt, guidance_scale=GUIDANCE_SCALE,
                 num_inference_steps=NUM_INFERENCE_STEPS).images[0]
    image = image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
    set_image(OUTPUT_IMG, image.copy())
    return image


def prompt_and_generate_button(prefix):
    prompt = st.text_area(
        "Prompt",
        value=DEFAULT_PROMPT,
        key=f"{prefix}-prompt",
    )
    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating your image, please wait..."):
            image = generate(prompt)
        st.image(image)


def txt2img_tab():
    prompt_and_generate_button("txt2img")


def settings_tab():
    global GUIDANCE_SCALE
    global DEFAULT_WIDTH
    global DEFAULT_HEIGHT
    global NUM_INFERENCE_STEPS
    DEFAULT_HEIGHT = st.slider("Final height", value=512, min_value=16, max_value=1024,
                               help="After getting an image there will be resizing for specified height.")
    DEFAULT_WIDTH = st.slider("Final width", value=512, min_value=16, max_value=1024,
                              help="After getting an image there will be resizing for specified width.")
    NUM_INFERENCE_STEPS = st.slider("Num inference steps", value=50, min_value=1, max_value=100,
                                    help="More steps usually lead to a higher quality of image and slower inference.")
    GUIDANCE_SCALE = st.slider("Guidance scale", value=7, min_value=0, max_value=30,
                               help="A higher hint scale encourages images that are closely related to the text hint, "
                                    "usually at the expense of lower image quality.")


def main():
    st.set_page_config(layout="wide")
    st.title("Stable Diffusion + Streamlit SPbU 3 Semester homework by [lastdesire](https://github.com/lastdesire)")
    tab, settings = st.tabs(
        ["Text to Image", "Settings"]
    )
    with settings:
        settings_tab()

    with tab:
        txt2img_tab()

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
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 7
    main()
