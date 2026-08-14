import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Live Face & Object Recognition", layout="wide")
st.title("🎯 Live Face & Object Recognition")
st.write("Recognizes known faces by name, and labels other people/objects generically.")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_resource
def train_recognizer():
    known_dir = "known_faces"
    cascade = load_face_cascade()
    images, labels, label_map = [], [], {}
    current_label = 0
    skipped = []

    if os.path.exists(known_dir):
        for person_name in sorted(os.listdir(known_dir)):
            person_dir = os.path.join(known_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            person_has_face = False
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None or img.size == 0:
                        skipped.append(img_name)
                        continue

                    faces = cascade.detectMultiScale(img, 1.1, 5)
                    for (x, y, w, h) in faces:
                        face_roi = img[y:y+h, x:x+w]
                        if face_roi.size == 0:
                            continue
                        face_roi = cv2.resize(face_roi, (200, 200))
                        images.append(face_roi)
                        labels.append(current_label)
                        person_has_face = True
                except Exception:
                    skipped.append(img_name)
                    continue

            if person_has_face:
                label_map[current_label] = person_name.capitalize()
                current_label += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if images:
        recognizer.train(images, np.array(labels))

    return recognizer, label_map, skipped

yolo_model = load_yolo()
face_cascade = load_face_cascade()
recognizer, label_map, skipped_files = train_recognizer()

if skipped_files:
    st.warning(f"⚠️ Skipped unreadable files during training: {', '.join(skipped_files)}")
if not label_map:
    st.warning("⚠️ No faces were learned. Check your known_faces folders and photos.")
else:
    st.info(f"✅ Learned faces for: {', '.join(label_map.values())}")

friendly_names = {
    0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus",
    7: "Truck", 15: "Cat", 16: "Dog", 24: "Backpack", 26: "Handbag",
    39: "Bottle", 41: "Cup", 63: "Laptop", 67: "Phone", 73: "Book"
}

CONFIDENCE_THRESHOLD = 70  # LBPH: lower = better/stricter match

def process_frame(frame):
    results = yolo_model(frame, conf=0.4, verbose=False)[0]
    counts = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = friendly_names.get(cls_id, "Object")

        if cls_id == 0:  # person -> try face recognition
            person_crop = frame[max(0, y1):y2, max(0, x1):x2]
            name = "Person"
            if person_crop.size > 0:
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (fx, fy, fw, fh) in faces:
                    face_roi = gray[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue
                    face_roi = cv2.resize(face_roi, (200, 200))
                    if len(label_map) > 0:
                        try:
                            pred_label, confidence = recognizer.predict(face_roi)
                            if confidence < CONFIDENCE_THRESHOLD:
                                name = label_map.get(pred_label, "Person")
                        except cv2.error:
                            pass
                    break
            label = name

        counts[label] = counts.get(label, 0) + 1
        color = (0, 255, 0) if label not in ("Person", "Object") else (255, 165, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, counts

mode = st.radio("Choose input source:", ["📁 Upload Video", "📷 Live Camera"])

if mode == "📁 Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        input_temp.write(uploaded_file.read())
        input_path = input_temp.name

        if st.button("▶️ Start Processing"):
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            max_width = 640
            if width > max_width:
                scale = max_width / width
                width, height = int(width * scale), int(height * scale)

            output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            progress_bar = st.progress(0, text="Processing video...")
            preview = st.empty()
            frame_idx = 0
            final_counts = {}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (width, height))
                annotated, counts = process_frame(frame)
                final_counts = counts
                writer.write(annotated)

                if frame_idx % 15 == 0:
                    preview.image(annotated, channels="BGR", use_container_width=True)

                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            writer.release()
            st.success("✅ Done!")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.video(output_path)

            st.subheader("📊 Detected (last frame)")
            if final_counts:
                cols = st.columns(len(final_counts))
                for i, (label, count) in enumerate(final_counts.items()):
                    cols[i].metric(label, count)

            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f, file_name="output.mp4")

else:
    st.write("Click 'START' below and allow camera access in your browser.")

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = process_frame(img)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(key="live", video_processor_factory=VideoProcessor)
