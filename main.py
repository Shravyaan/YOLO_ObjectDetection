import streamlit as st
from ultralytics import YOLO
from video_detection import run_video_detection

from image_detection import run_image_detection

st.set_page_config(page_title="YOLO model by Shravya", layout="wide")
st.title("YOLO Detection App")

model_path = "yolov8n.pt" 
model = YOLO(model_path)

option = st.sidebar.selectbox(
    "Select Mode",
    ["Evaluate Image", "Evaluate video"]
)

if option == "Evaluate Image":
    run_image_detection(model)
elif option == "Evaluate video":
    run_video_detection(model)



# to run this type 
# python -m streamlit run main.py 