import cv2
import streamlit as st
from ultralytics import YOLO

from face_module import detect_faces
from vehicle_module import detect_vehicles
from tracking import update_tracks

# ---------------- UI ----------------
st.title("🤖 AI Surveillance System")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()

# ---------------- SESSION STATE ----------------
if "vehicle_tracks" not in st.session_state:
    st.session_state.vehicle_tracks = {}

if "vehicle_count" not in st.session_state:
    st.session_state.vehicle_count = 0

if "track_id" not in st.session_state:
    st.session_state.track_id = 0

if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    faceNet = cv2.dnn.readNet(
        "opencv_face_detector_uint8.pb",
        "opencv_face_detector.pbtxt"
    )
    yolo = YOLO("yolov8n.pt")
    return faceNet, yolo

faceNet, yolo = load_models()

line_y = 300
cap = st.session_state.cap

# ---------------- MAIN ----------------
if run:
    ret, frame = cap.read()

    if not ret:
        st.error("Camera Error")
    else:
        # ---------------- PREPROCESS ----------------
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=-20)

        # ---------------- FACE ----------------
        faces = detect_faces(frame, faceNet)
        for x1, y1, x2, y2 in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------------- VEHICLE ----------------
        vehicles = detect_vehicles(frame, yolo)
        for x1, y1, x2, y2, name in vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ---------------- TRACKING ----------------
        st.session_state.vehicle_tracks, \
        st.session_state.track_id, \
        st.session_state.vehicle_count = update_tracks(
            vehicles,
            st.session_state.vehicle_tracks,
            st.session_state.track_id,
            line_y,
            st.session_state.vehicle_count
        )

        # ---------------- LINE ----------------
        cv2.line(frame, (0, line_y),
                 (frame.shape[1], line_y), (0, 0, 255), 2)

        # ---------------- COUNT DISPLAY ----------------
        cv2.putText(frame,
                    f"Count: {st.session_state.vehicle_count}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

        # ---------------- SHOW ----------------
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # 🔁 Auto-refresh (IMPORTANT)
        st.rerun()

# ---------------- STOP CAMERA ----------------
else:
    if "cap" in st.session_state:
        st.session_state.cap.release()