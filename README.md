# Stable diffusion with streamlit

This is a homework assignment for a computer workshop in the third semester (SPbU SE) related to neural networks.
## About
In this project, we implemented the streamlit application for working with neural networks (stable diffusion for generating images from text entered by the user).
Note that image generation in this project takes place using a CPU.

## Usage
To start working with the application, first install all the necessary dependencies:
```
pip install -r requirements.txt
```
Next, you need to accept the agreement located at this link: https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main (you must be registered). After that, you should get an access token (write) in your personal profile. Now run the command: 
```
huggingface-cli login
```
and insert your token.

That's almost all!

You are ready to run application:
```
streamlit run src/main.py
```

## References to materials
stable-diffusion: https://huggingface.co/runwayml/stable-diffusion-v1-5

streamlit: https://streamlit.io/
