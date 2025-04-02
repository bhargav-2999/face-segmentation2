import streamlit as st
import cv2
import numpy as np
from PIL import Image
from face_segment_fin import process_image, detect_single_face

st.set_page_config(page_title="Face Segmentation App", layout="centered")

st.title("🧠 Face Segmentation App")
st.markdown("Upload an image with **exactly one face** to get a segmented face with a transparent background.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    if not detect_single_face(image_rgb):
        st.error("Please upload an image with **exactly ONE face**.")
    else:
        with st.spinner("Processing... Please wait."):
            try:
                rgba_image = process_image(image_bgr)
                output_pil = Image.fromarray(rgba_image)

                st.success("Face segmentation successful!")
                st.image(output_pil, caption="Segmented Image (Transparent BG)", use_column_width=True)

                st.download_button(
                    label="Download PNG",
                    data=cv2.imencode('.png', rgba_image)[1].tobytes(),
                    file_name="segmented_face.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error during processing: {e}")
