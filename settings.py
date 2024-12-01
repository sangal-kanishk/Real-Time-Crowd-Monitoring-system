from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
VIDEO = 'Video'
WEBCAM = 'Webcam'
LIVESTREAM = 'Live Stream'

SOURCES_LIST = [VIDEO, WEBCAM, LIVESTREAM]
print (ROOT)
# ML Model config
MODEL_DIR = ROOT / 'weights'
#YOLOV8N = MODEL_DIR / 'yolov8n.pt'
YOLOV8N = 'weights /yolov8n.pt'
YOLOV8S = MODEL_DIR / 'yolov8s.pt'
YOLOV8M = MODEL_DIR / 'yolov8m.pt'
YOLOV8L = MODEL_DIR / 'yolov8l.pt'
YOLOV8X = MODEL_DIR / 'yolov8x.pt'
# In case of your custom model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'
'''
if not MODEL_DIR.exists():
    print("Model directory does not exist:", MODEL_DIR)
# Print the model directory path
print("Model Directory:", MODEL_DIR)

#-----------------------------------

for model_file in [YOLOV8N, YOLOV8S, YOLOV8M, YOLOV8L, YOLOV8X]:
    if not model_file.exists():
        print(f"Model file does not exist: {model_file}")
    else:
        print(f"Model file exists: {model_file}")
'''
#---------------------------------



# Set the class(es) to be detected
PERSON = 0
CAR = 2

CLASSES_DICT = {
    'Person': PERSON,
    'Car': CAR
}

CLASSES_LIST = [PERSON, CAR]

# Webcam
WEBCAM_PATH = 0