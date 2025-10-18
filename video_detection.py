import streamlit as st
import cv2
import numpy as np
import tempfile
import imageio

def run_video_detection(model):
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        tempof = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tempof.write(uploaded_file.read())
        tempof.close()
        video_path = tempof.name

        if st.button("Decode video objects"):
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_file.close()
            writer = imageio.get_writer(out_file.name, fps=fps)

            progress_bar = st.progress(0)
            total_frames = reader.count_frames()

            for i, frame in enumerate(reader):
                frame_resized = cv2.resize(frame, (640, 640))
                results = model.predict(frame_resized, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
                writer.append_data(annotated_frame)
                progress_bar.progress(min((i + 1) / total_frames, 1.0))

            writer.close()
            st.video(out_file.name)