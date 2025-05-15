import streamlit as st
import detect_and_read
from PIL import Image
import pandas as pd
import torch
import os


def main():
    st.title("License Plate Recognition")
    with st.sidebar:
        st.header("⚙️ Configuration")
        confidence = st.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
        use_cuda = st.checkbox("Use CUDA (GPU)", value=True)
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and device == "cpu":
            st.warning("CUDA requested but not available. Running on CPU.")
        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type
        df = pd.DataFrame()

        if "image" in file_type:
            image = Image.open(uploaded_file).convert("RGB")
            result_img, df = detect_and_read.detect_and_read_image(image, conf_thresh=confidence, device=device)
            st.image(result_img, caption="Detected Image", use_container_width=True)

        elif "video" in file_type:
            st.video(uploaded_file)
            with st.spinner("Processing video..."):
                uploaded_file.seek(0)
                output_video_path, df = detect_and_read.detect_and_read_video(uploaded_file, conf_thresh=confidence,
                device=device)
            if os.path.exists(output_video_path):
                st.video(output_video_path)
                os.remove(output_video_path)  # Clean up
            else:
                st.error("Failed to generate annotated video.")

        with st.sidebar:
            if not df.empty:
                st.subheader("Detected Plates")
                st.write(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "plates.csv", "text/csv")
            else:
                st.write("No license plates detected.")


if __name__ == "__main__":
    main()