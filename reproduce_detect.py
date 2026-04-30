from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

VIDEO_PATH = "test1.mp4"
MODEL_PATH = "best-7.pt"
CONF_THRESHOLD = 0.15
stride = 7

model = YOLO(MODEL_PATH)
results = model.predict(source=VIDEO_PATH, stream=True, vid_stride=stride, conf=CONF_THRESHOLD, imgsz=1280, verbose=False)

all_detections = []
for i, result in enumerate(results):
    if result.boxes is not None:
        for box in result.boxes:
            all_detections.append({
                'frame': i * stride,
                'class_name': model.names[int(box.cls[0])],
                'conf': float(box.conf[0])
            })

df = pd.DataFrame(all_detections)
if not df.empty:
    print(df['class_name'].value_counts())
    print("\nGoal detections:")
    print(df[df['class_name'] == 'goal'].head())
else:
    print("No detections at all!")
