import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from copy import deepcopy

from src.core import process_inpaint


def image_download_button(pil_image, filename: str, fmt: str, label="Download"):
    if fmt not in ["jpg", "png"]:
        raise Exception(f"Unknown image format (Available: {fmt} - case sensitive)")
    
    pil_format = "JPEG" if fmt == "jpg" else "PNG"
    file_format = "jpg" if fmt == "jpg" else "png"
    mime = "image/jpeg" if fmt == "jpg" else "image/png"
    
    buf = BytesIO()
    pil_image.save(buf, format=pil_format)
    
    return st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=f'{filename}.{file_format}',
        mime=mime,
    )

if "button_id" not in st.session_state:
    st.session_state["button_id"] = ""
if "color_to_label" not in st.session_state:
    st.session_state["color_to_label"] = {}

if 'reuse_image' not in st.session_state:
    st.session_state.reuse_image = None
def set_image(img):
    st.session_state.reuse_image = img

st.title("Magic-Eraser")

st.image(open("assets/demo.png", "rb").read())

st.markdown(
    """
    You don't have to worry about mastering photo editing techniques to remove an object from your photo. **Simply mark over the areas you want to erase, and our AI will take care of the rest.**
    """
)
uploaded_file = st.file_uploader("Choose image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    if st.session_state.reuse_image is not None:
        img_input = Image.fromarray(st.session_state.reuse_image)
    else:
        bytes_data = uploaded_file.getvalue()
        img_input = Image.open(BytesIO(bytes_data)).convert("RGBA")

 #resize
    max_size = 2000
    img_width, img_height = img_input.size
    if img_width > max_size or img_height > max_size:
        if img_width > img_height:
            new_width = max_size
            new_height = int((max_size / img_width) * img_height)
        else:
            new_height = max_size
            new_width = int((max_size / img_height) * img_width)
        img_input = img_input.resize((new_width, new_height))
    
    stroke_width = st.slider("Brush size", 1, 100, 50)

    st.write("**Now draw (brush) the part of image that you want to remove.**")
    
    canvas_bg = deepcopy(img_input)
    aspect_ratio = canvas_bg.width / canvas_bg.height
    streamlit_width = 720
    
    if canvas_bg.width > streamlit_width:
        canvas_bg = canvas_bg.resize((streamlit_width, int(streamlit_width / aspect_ratio)))
    
    canvas_result = st_canvas(
        stroke_color="rgba(255, 0, 255, 1)",
        stroke_width=stroke_width,
        background_image=canvas_bg,
        width=canvas_bg.width,
        height=canvas_bg.height,
        drawing_mode="freedraw",
        key="compute_arc_length", 
    )
    
    if canvas_result.image_data is not None:
        im = np.array(Image.fromarray(canvas_result.image_data.astype(np.uint8)).resize(img_input.size))
        background = np.where(
            (im[:, :, 0] == 0) & 
            (im[:, :, 1] == 0) & 
            (im[:, :, 2] == 0)
        )
        drawing = np.where(
            (im[:, :, 0] == 255) & 
            (im[:, :, 1] == 0) & 
            (im[:, :, 2] == 255)
        )
        im[background]=[0,0,0,255]
        im[drawing]=[0,0,0,0] # RGBA
        
        reuse = False
        
        if st.button('Submit'):
            
            with st.spinner("AI is doing the magic!"):
                output = process_inpaint(np.array(img_input), np.array(im)) #TODO Put button here
                img_output = Image.fromarray(output).convert("RGB")
            
            st.write("AI has finished the job!")
            st.image(img_output)
            # reuse = st.button('Edit again (Re-use this image)', on_click=set_image, args=(inpainted_img, ))
            
            uploaded_name = os.path.splitext(uploaded_file.name)[0]
            image_download_button(
                pil_image=img_output,
                filename=uploaded_name,
                fmt="jpg",
                label="Download Image"
            )
            
            st.info("**TIP**: If the result is not perfect, you can download it then "
                    "upload then remove the artifacts.")
