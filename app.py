import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# ---------------- UI ----------------
st.set_page_config(page_title="Smart AI Surveillance + Traffic", layout="wide")
st.title("🤖 AI Surveillance + 🚗 Traffic System")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

# ---------------- SESSION STATE (FIXED GLOBAL ISSUE) ----------------
if "vehicle_tracks" not in st.session_state:
    st.session_state.vehicle_tracks = {}

if "vehicle_count" not in st.session_state:
    st.session_state.vehicle_count = 0

if "track_id" not in st.session_state:
    st.session_state.track_id = 0

# ---------------- UI METRICS ----------------
col1, col2, col3, col4, col5 = st.columns(5)
face_box = col1.empty()
male_box = col2.empty()
female_box = col3.empty()
status_box = col4.empty()
vehicle_box = col5.empty()

# ---------------- MODELS ----------------
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

yolo = YOLO("yolov8n.pt")

MODEL_MEAN_VALUES = (78.426, 87.768, 114.895)

ageList = ['(0-5)','(5-10)','(10-15)','(15-20)','(20-25)','(25-30)',
           '(30-35)','(35-40)','(40-45)','(45-50)']
genderList = ['Male','Female']

# ---------------- VARIABLES ----------------
line_y = 300
prev_frame = None
frame_count = 0
prev_time = 0
movement_threshold = 400000

# ---------------- FACE DETECTION ----------------
def faceBox(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > 0.7:
            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)
            boxes.append([x1,y1,x2,y2])
    return boxes

# ---------------- VEHICLE DETECTION ----------------
def detect_vehicles(frame):
    results = yolo(frame)
    detections = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            name = yolo.names[cls]

            if name in ["car","motorbike","bus","truck"]:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                detections.append((x1,y1,x2,y2,name))

    return detections

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera Error")
        break

    frame = cv2.resize(frame, (640,480))
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)

    # ---------------- MOVEMENT ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray
        continue

    diff = cv2.absdiff(prev_frame, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    movement = np.sum(thresh)
    prev_frame = gray

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # ---------------- FACE ----------------
    small = cv2.resize(frame, (300,300))
    faces = faceBox(small)

    sx = frame.shape[1] / 300
    sy = frame.shape[0] / 300

    faces = [[int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)] for x1,y1,x2,y2 in faces]

    male = 0
    female = 0
    status = "Idle"

    # ---------------- FACE LOOP ----------------
    for x1,y1,x2,y2 in faces:
        face = frame[max(0,y1-20):min(y2+20,frame.shape[0]),
                     max(0,x1-20):min(x2+20,frame.shape[1])]

        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)

        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        if gender == "Male":
            male += 1
        else:
            female += 1

        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        cx = (x1+x2)//2
        if cx < frame.shape[1]*0.4:
            att = "Left"
        elif cx > frame.shape[1]*0.6:
            att = "Right"
        else:
            att = "Center"

        if movement > movement_threshold:
            status = "Active"
        elif att != "Center":
            status = "Distracted"
        else:
            status = "Focused"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"{gender},{age},{status}",
                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # ---------------- VEHICLES ----------------
    vehicles = detect_vehicles(frame)

    new_tracks = {}

    for x1,y1,x2,y2,name in vehicles:
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        matched = None

        for tid,(px,py,done) in st.session_state.vehicle_tracks.items():
            dist = np.sqrt((cx-px)**2 + (cy-py)**2)
            if dist < 60:
                matched = tid
                break

        if matched is None:
            matched = f"id_{st.session_state.track_id}"
            st.session_state.track_id += 1

        prev_done = st.session_state.vehicle_tracks.get(matched,(0,0,False))[2]

        if cy > line_y and not prev_done:
            st.session_state.vehicle_count += 1
            prev_done = True

        new_tracks[matched] = (cx,cy,prev_done)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,name,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    st.session_state.vehicle_tracks = new_tracks

    # ---------------- LINE ----------------
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),2)

    # ---------------- UI ----------------
    face_box.metric("Faces",len(faces))
    male_box.metric("Male",male)
    female_box.metric("Female",female)
    status_box.metric("Status",status)
    vehicle_box.metric("Vehicles",st.session_state.vehicle_count)

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()