# Python In-built packages
from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Capstone Project - Object Detection Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Read the class names from coco.txt (change it using a custom yaml file while training a custom model)
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Defne the class options
class_options = {
    0: "Person",
    2: "Car"
}

# Main page heading
st.title("Capstone Project - Object Detection Dashboard")

# Sidebar
st.sidebar.header("CV Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting the model to be used for detection task
if model_type == 'YOLOv8n':
    model_path = Path(settings.YOLOV8N).resolve()
    print("Model Path:", model_path)
    
elif model_type == 'YOLOv8s':
    model_path = Path(settings.YOLOV8S)
if model_type == 'YOLOv8m':
    model_path = Path(settings.YOLOV8M)
elif model_type == 'YOLOv8l':
    model_path = Path(settings.YOLOV8L)
if model_type == 'YOLOv8x':
    model_path = Path(settings.YOLOV8X)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
    print("Model loaded")
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Define the header of the options
st.sidebar.header("Class Selection")

# Select the classes to be loaded
classes_to_detect = st.sidebar.multiselect(
    "Select the class(es) to detect",
    settings.CLASSES_LIST,
    [0],
    format_func = lambda x : class_list[x] #class_options.get(x)
)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source :sunglasses:", settings.SOURCES_LIST)

if source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, classes_to_detect)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model, classes_to_detect)
    
elif source_radio == settings.LIVESTREAM:
    helper.play_rtsp_stream(confidence, model, classes_to_detect)

else:
    st.error("Please select a valid source type!")