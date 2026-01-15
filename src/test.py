from ultralytics import YOLO  # type: ignore
import ultralytics.engine.results
import cv2
import numpy as np
import torch
from src.models.clip import CLIP
import os

clip = CLIP()
model = YOLO(r"temp\detect\train_yolo11n\weights\best.pt")
p = "data/truncated_videos/0000.mp4"

cap = cv2.VideoCapture(p)
cap2 = cv2.VideoWriter(
    "temp/export.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),  # type: ignore
    30,
    (960, 540)
)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, conf=0.4, iou=0.8, tracker="botsort.yaml")
    result = results[0]
    if result.boxes is None:
        continue
    data = result.boxes.data
    for i in range(data.shape[0]):
        data0 = data[i].reshape(1, -1)
        boxes = ultralytics.engine.results.Boxes(data0, result.boxes.orig_shape)
        result.boxes = boxes
        line_weight = 2
        frame = result.plot(img=frame, line_width=line_weight)
    annotated_frame = cv2.resize(frame, (960, 540))


    cap2.write(annotated_frame)
    cv2.imshow("Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cap2.release()
cv2.destroyAllWindows()