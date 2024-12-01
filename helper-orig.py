#from deep_sort.tracker import Tracker
from ultralytics import YOLO
from plyer import notification 
import numpy as np
import os
import math
import time
import torch
import tempfile
import datetime
import streamlit as st
import cv2
import pandas as pd
import cvzone
from tracker import*
from fpdf import FPDF
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from itertools import zip_longest
# from pytube import YouTube

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to("cpu")
    # model = YOLO(model_path).to(device)  # Uncomment this line if running on GPU and comment the line above
    
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def send_notification(message):
    """ Sends a desktop notification """
    notification.notify(
        title="People Detection Alert",
        message=message,
        timeout=10
    )

def detect_people_and_notify(frame, model, tracker, total_count, threshold=5):
    """
    Detects people in a frame and triggers notification if people count exceeds threshold.

    Parameters:
        frame: The frame from the video or webcam stream.
        model: The YOLO model instance.
        tracker: Instance of the Tracker.
        total_count: List tracking total number of people per frame.
        threshold: The number of people exceeding which the notification is triggered.
    
    Returns:
        Updated frame with detections.
        Number of people detected in the frame.
    """
    # YOLO model inference
    results = model(frame)
    
    # Detect people in the frame
    bbox_id = tracker.update(results[0].boxes.data)
    
    # Number of people detected in the frame
    num_people_in_frame = len(bbox_id)
    
    # Append the total count
    total_count.append(num_people_in_frame)
    
    # Notification Trigger: Check if the count exceeds the threshold (15 people)
    if num_people_in_frame > threshold:
        send_notification(f"Warning: {num_people_in_frame} people detected in the frame, exceeding the threshold of {threshold}.")
    
    # Display detected people and counts on the frame
    cvzone.putTextRect(frame, f'Total: {num_people_in_frame}', (700, 60), 2, 2)
    
    return results[0].plot(), num_people_in_frame

def play_webcam(conf, model, classes_to_detect, threshold=15):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    placeholder = st.empty()

    # Create a list to store the total number of people in each frame
    total_count = []

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    count = 0
    persondown = {}
    tracker = Tracker()
    counter1 = []
    personup = {}
    counter2 = []
    cy1 = 194
    cy2 = 220
    offset = 6
    
    frame_save = st.sidebar.radio(
        "Do you want to save the frames with detections?", ["No", "Yes"]
    )
    
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        
        cap = cv2.VideoCapture(source_webcam)
        
        if frame_save == "Yes":

            # Creating a directory for saving the reports
            if not os.path.exists('./Assets'):
                os.makedirs('./Assets')

            if not os.path.exists('./Assets/Reports'):
                os.makedirs('./Assets/Reports')

            rprt_folderpath = os.path.join('Assets', 'Reports')
            rprt_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            rprt_ts_folderpath = os.path.join(rprt_folderpath, rprt_timestamp)
            os.makedirs(rprt_ts_folderpath)

            if not os.path.exists('./Assets/Detected_Frames'):
                os.makedirs('./Assets/Detected_Frames')

            common_folderpath = os.path.join('Assets', 'Detected_Frames')
            
            if not os.path.exists('./Assets/Detected_Frames/CCTV'):
                os.makedirs('./Assets/Detected_Frames/CCTV')
            
            frame_folderpath = os.path.join(common_folderpath, 'CCTV')
            
            fs_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            frames_folder = os.path.join(frame_folderpath, fs_timestamp)
            os.makedirs(frames_folder)
        
        report = []
        st_frame = st.empty()
        
        while True:    
            ret,frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf, classes=classes_to_detect)
            a = results[0].boxes.data.cpu().numpy()
            px = pd.DataFrame(a).astype("float")
            list = []
        
            for index,row in px.iterrows():
        
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                
                c = class_list[d]
                if 'person' in c:
                    list.append([x1, y1, x2, y2])

            bbox_id = tracker.update(list)
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2
                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                    
            ######  For people moving down (entering the facility)
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                persondown[id] = (cx, cy)
                
            if id in persondown:
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if frame_save == "Yes":
                        results_frame = results[0].plot()
                        entry_folder = os.path.join(frames_folder, "Entry")
                        if not os.path.exists(entry_folder):
                            os.makedirs(entry_folder)
                        name = f"frame_{frame_number:05d}.jpg"
                        frame_filename = os.path.join(entry_folder, name)
                        image = np.array(results_frame)
                        cv2.imwrite(frame_filename, image)
                    if counter1.count(id) == 0:
                        counter1.append(id)

            ######  For people moving up (exiting the facility)
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                personup[id] = (cx, cy)
                
            if id in personup:
                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if frame_save == "Yes":
                        results_frame = results[0].plot()
                        exit_folder = os.path.join(frames_folder, "Exit")
                        if not os.path.exists(exit_folder):
                            os.makedirs(exit_folder)
                        name = f"frame_{frame_number:05d}.jpg"
                        frame_filename = os.path.join(exit_folder, name)
                        image = np.array(results_frame)
                        cv2.imwrite(frame_filename, image)
                    if counter2.count(id) == 0:
                        counter2.append(id)
                
            cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
            cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)
        
            down = len(counter1)
            up = len(counter2)
            
            count = len(bbox_id)
            total_count.append(len(bbox_id))
            
            res_plotted = results[0].plot()
            st_frame.image(res_plotted,
                        caption='Detected Video',
                        width=1000,
                        channels="BGR", 
                        use_column_width=False
                        )
            
            placeholder.markdown(f"<p style'color:white;'>Number of people going up: {up}</p>", unsafe_allow_html=True)
            placeholder.markdown(f"<p style'color:white;'>Number of people going down: {down}</p>", unsafe_allow_html=True)
            placeholder.markdown(f"<p style'color:white;'>The total number of people in the frame are: {len(bbox_id)}</p>", unsafe_allow_html=True)
            
            # Trigger notification if people count exceeds threshold
            if len(bbox_id) > threshold:
                print(f"Notification triggered: {len(bbox_id)} people detected.")
                send_notification(f"Warning: {len(bbox_id)} people detected, exceeding the threshold of {threshold}.")
            
            # Creating a report of the detection and corresponding timestamp
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = frame_number / fps
            timestamp_str = str(datetime.timedelta(seconds=timestamp))
            report.append([timestamp_str, count])
        
        cap.release()
        placeholder.text_area(f"\nThe maximum number of people captured in a single frame are: {max(total_count)}\n")




def play_stored_video(conf, model, classes_to_detect, threshold=15):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        threshold: The threshold for triggering notifications (default set to 15).

    Returns:
        None

    Raises:
        None
    """

    # Create a list to store the total number of people in each frame
    total_count = []

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    
    print(class_list)

    persondown = {}
    tracker = Tracker()
    counter1 = []
    personup = {}
    counter2 = []
    cy1 = 194
    cy2 = 220
    offset = 6
    fps_tracker = 0
    
    source_vid = st.sidebar.file_uploader(
        label="Upload a video."
    )
    
    fps_set = float(st.sidebar.slider(
        f"Select processing speed of video. \n\n\nWARNING! It may result in a slight loss of information", 1, 1, 5
    ))
    
    frame_save = st.sidebar.radio(
        "Would you like to save the frames with detections along with detection report?", ["No", "Yes"]
    )
    
    placeholder = st.empty()

    if st.sidebar.button('Detect Video Objects'):
            
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(source_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st_frame = st.empty()
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        time_placeholder = st.empty()
        
        start_time = time.time()
        
        if frame_save == "Yes":
            if not os.path.exists('./Assets'):
                os.makedirs('./Assets')

            if not os.path.exists('./Assets/Reports'):
                os.makedirs('./Assets/Reports')

            rprt_folderpath = os.path.join('Assets', 'Reports')
            rprt_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            rprt_ts_folderpath = os.path.join(rprt_folderpath, rprt_timestamp)
            os.makedirs(rprt_ts_folderpath)

            if not os.path.exists('./Assets/Detected_Frames'):
                os.makedirs('./Assets/Detected_Frames')

            common_folderpath = os.path.join('Assets', 'Detected_Frames')
            if not os.path.exists('./Assets/Detected_Frames/Video'):
                os.makedirs('./Assets/Detected_Frames/Video')

            frames_folder = os.path.join(common_folderpath, rprt_timestamp)
            os.makedirs(frames_folder)

        report = []
        
        while True:    
            ret, frame = cap.read()
            if not ret:
                break
            
            fps_tracker += 1
            if fps_tracker % fps_set != 0:
                continue
            
            results = model.predict(frame, conf=conf, classes=classes_to_detect)
            a = results[0].boxes.data.cpu().numpy()
            px = pd.DataFrame(a, columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class']).astype("float")
            list = []
        
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                
                c = class_list[d]
                if 'person' in c:
                    list.append([x1, y1, x2, y2])

            bbox_id = tracker.update(list)
            total_count.append(len(bbox_id))
            
            # Notification trigger: If number of people exceeds threshold
            if len(bbox_id) > threshold:
                print(f"Notification triggered: {len(bbox_id)} people detected.")
                send_notification(f"Warning: {len(bbox_id)} people detected in the video frame, exceeding the threshold of {threshold}.")
            
            res_plotted = results[0].plot()
            st_frame.image(res_plotted, caption='Detected Video', width=1000, channels="BGR", use_column_width=False)
            
            placeholder.markdown(f"<p style='color:white;'>The total number of people in the frame are: {len(bbox_id)}</p>", unsafe_allow_html=True)

        cap.release()
        
        # Handle possible empty total_count
        if total_count:
            max_count = max(total_count)
        else:
            max_count = 0

        placeholder.text_area(f"\nThe maximum number of people captured in a single frame are: {max_count}\n")
'''
def play_rtsp_stream(conf, model, classes_to_detect):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    
    placeholder = st.empty()

    # Create a list to store the total number of people in each frame
    total_count = []

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 
    #print(class_list)

    count = 0
    persondown = {}
    #tracker = Tracker()
    counter1 = []

    personup = {}
    counter2 = []
    cy1 = 194
    cy2 = 220
    offset = 6
    
    frame_save = st.sidebar.radio(
        "Do you want to save the frames with detections?", ["No", "Yes"]
    )
    
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    
    if st.sidebar.button('Detect Objects'):
        # try:
            cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_rtsp)))
            st_frame = st.empty()
            while True:    
                ret,frame = cap.read()
                if not ret:
                    break
            #    frame = stream.read()

                count += 1
                if count % 3 != 0:
                    continue
                frame=cv2.resize(frame,(1020,500))
            

                results = model.predict(frame, conf=conf, classes=classes_to_detect)
            #   print(results)
                a = results[0].boxes.data
                px = pd.DataFrame(a).astype("float")
            #    print(px)
                list = []
            
                for index,row in px.iterrows():
            #        print(row)
            
                    x1 = int(row[0])
                    y1 = int(row[1])
                    x2 = int(row[2])
                    y2 = int(row[3])
                    d = int(row[5])
                    
                    c = class_list[d]
                    if 'person' in c:

                        list.append([x1,y1,x2,y2])
                
                
                bbox_id = tracker.update(list)
                for bbox in bbox_id:
                    x3,y3,x4,y4,id = bbox
                    cx = int(x3+x4)//2
                    cy = int(y3+y4)//2
                    cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
                    
            ######  For people moving down      
                    
                    if cy1<(cy+offset) and cy1>(cy-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                        cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        persondown[id] = (cx,cy)
                        
                    if id in persondown:
                        if cy2<(cy+offset) and cy2>(cy-offset):
                            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                            cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                            if counter1.count(id) == 0:
                                counter1.append(id)
            
            ######  For people moving up
                    
                    if cy2<(cy+offset) and cy2>(cy-offset):
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
                        cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                        personup[id] = (cx,cy)
                        
                    if id in personup:
                        if cy1<(cy+offset) and cy1>(cy-offset):
                            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)
                            cvzone.putTextRect(frame, f'{id}',(x3,y3),1,2)
                            if counter2.count(id) == 0:
                                counter2.append(id)       
                    
                cv2.line(frame,(3,cy1),(1018,cy1),(0,255,0),2)
                cv2.line(frame,(5,cy2),(1019,cy2),(0,255,255),2)
            
                down = len(counter1)
                up = len(counter2)
                
                cvzone.putTextRect(frame, f'Down: {down}',(50,60),2,2)
                cvzone.putTextRect(frame, f'Up: {up}',(50,160),2,2)
                cvzone.putTextRect(frame, f'Total: {len(bbox_id)}',(700,60),2,2)
                
                total_count.append(len(bbox_id))
                res_plotted = results[0].plot()
                st_frame.image(res_plotted,
                            caption='Detected Video',
                            channels="BGR", 
                            use_column_width=True
                            )
                
                placeholder.text(f"\nNumber of people going up: {up}")
                placeholder.text(f"\nNumber of people going down: {down}")
                placeholder.text(f"\nThe total number of people in the frame are: {len(bbox_id)}")
                
                if frame_save == "Yes":
                    if len(results[0]) >= 1:
                        results_frame = results[0].plot()  
                        name="frames/frame_%d.jpg" % count
                        image = np.array(results_frame)
                        cv2.imwrite(name, image)
                        count+=1
                
            cap.release()
            st.write(f"\nThe maximum number of people captured in a single frame are: {max(total_count)}\n")
            '''