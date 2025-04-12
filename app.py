# app.py
import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import os
import plotly.express as px
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set up page
st.set_page_config(page_title="📦 Real-Time Object Detection + Dashboard", layout="wide")
st.title("📦 Real-Time Object Detection + Dashboard")

# File paths
CSV_PATH = "object_log.csv"
MODEL_PATH = "yolov8n.pt"

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize object log
if not os.path.exists(CSV_PATH):
    df_init = pd.DataFrame(columns=["Object", "Confidence", "Distance", "Date", "Time"])
    df_init.to_csv(CSV_PATH, index=False)

# Helper: Log Detection
def log_detection(object_name, confidence, distance=0):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    df = pd.DataFrame([[object_name, confidence, distance, date_str, time_str]],
                      columns=["Object", "Confidence", "Distance", "Date", "Time"])
    df.to_csv(CSV_PATH, mode='a', header=False, index=False)

# --- Section 1: Live Video Detection ---
st.subheader("🎥 Real-Time Object Detection")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = model(image)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = round(float(box.conf[0]), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            log_detection(label, conf)

        return image

webrtc_streamer(key="object-detect", video_transformer_factory=VideoTransformer)

# --- Section 2: Dashboard ---
st.subheader("📊 Detection Dashboard")

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time

    st.write("### 📝 Latest Detections")
    st.dataframe(df.tail(10), use_container_width=True)

    # 1. Total count
    st.metric("Total Detections", len(df))

    # 2. Object counts
    st.write("### 📦 Object-wise Count")
    obj_counts = df['Object'].value_counts()
    st.bar_chart(obj_counts)

    # 3. Detections over time
    st.write("### ⏱️ Detections Over Time")
    time_group = df.groupby("Time")["Object"].count()
    st.line_chart(time_group)

    # 4. Confidence by Object
    st.write("### 🔍 Confidence Levels")
    conf_chart = df.groupby("Object")["Confidence"].mean().reset_index()
    fig1 = px.bar(conf_chart, x="Object", y="Confidence", color="Object", title="Average Confidence")
    st.plotly_chart(fig1, use_container_width=True)

    # 5. Pie chart
    st.write("### 🥧 Object Distribution")
    fig2 = px.pie(df, names='Object', title='Detected Object Share')
    st.plotly_chart(fig2, use_container_width=True)

    # 6. Hourly Analysis
    st.write("### 🕒 Hourly Detection Pattern")
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
    hourly = df.groupby("Hour")["Object"].count().reset_index()
    fig3 = px.bar(hourly, x='Hour', y='Object', title="Detections per Hour")
    st.plotly_chart(fig3, use_container_width=True)

    # 7. Download button
    st.write("### 📥 Download Detection Logs")
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "object_log.csv", "text/csv")

else:
    st.warning("⚠️ No detection logs yet. Start the camera to log objects.")

