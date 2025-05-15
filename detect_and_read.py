from ultralytics import YOLO
import easyocr
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
import pandas as pd
import time
from collections import defaultdict, Counter

model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

def preprocess_plate(plate_img):
    """Basic preprocessing: grayscale and threshold"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_and_read_image(pil_image, conf_thresh=0.25, device="cpu"):
    """Detect and read license plates from an image"""
    img = np.array(pil_image)
    model.to(device)
    results = model.predict(img, conf=conf_thresh)[0]

    data = []
    plate_id = 1  # Start ID counter
    for box in results.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        cropped = img[ymin:ymax, xmin:xmax]
        processed = preprocess_plate(cropped)
        text = reader.readtext(processed, detail=0)
        plate_text = " ".join(text).strip()
        if plate_text:
            data.append({
                "id": plate_id,
                "text": plate_text
            })
            plate_id += 1

    result_img = results.plot()
    return result_img, pd.DataFrame(data)

def detect_and_read_video(video_file, conf_thresh=0.25, device="cpu"):
    """Detect and read license plates from a video, with duplicates in video but unique in CSV"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file.close()
    temp_path = temp_file.name

    output_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = output_video_file.name
    output_video_file.close()

    cap = cv2.VideoCapture(temp_path)
    annotated_frames = []
    plate_history = defaultdict(list)  # Track plates by location for frequency

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model.to(device)
        results = model.predict(frame, conf=conf_thresh)

        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                cropped = frame[ymin:ymax, xmin:xmax]
                processed = preprocess_plate(cropped)
                text = reader.readtext(processed, detail=0)
                plate_text = " ".join(text).strip()

                x_center = (xmin + xmax) // 2
                y_center = (ymin + ymax) // 2
                location_key = (x_center // 50, y_center // 50)

                if plate_text and len(plate_text) >= 3:
                    plate_history[location_key].append(plate_text)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frames.append(rgb_frame)

    cap.release()

    if annotated_frames:
        create_output_video(annotated_frames, output_video_path, fps=fps)
    else:
        print("Skipping video creation due to empty frames list.")

    time.sleep(0.5)
    os.remove(temp_path)

    # Aggregate unique plate texts for CSV, with an ID
    final_plates = []
    seen_plates = set()  # Track unique plate texts across all locations
    plate_id = 1  # Start ID counter
    for loc, texts in plate_history.items():
        counter = Counter(texts)
        for plate_text, count in counter.items():
            if plate_text not in seen_plates and count >= 2:  # Only unique plates with count >= 2
                seen_plates.add(plate_text)
                final_plates.append({
                    "id": plate_id,
                    "text": plate_text
                })
                plate_id += 1

    final_df = pd.DataFrame(final_plates)

    return output_video_path, final_df

def create_output_video(frames, output_path, fps=30):
    """Create a video from a list of frames"""
    if not frames:
        print("Error: No frames to write to video.")
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        return None

    for frame in frames:
        try:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for video writing
        except Exception as e:
            print(f"Error writing frame: {e}")
            out.release()
            return None

    out.release()
    cv2.destroyAllWindows()
    print(f"Video successfully written to {output_path}")
    return output_path