import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

model = YOLO(r"C:\Users\Om sinha\Downloads\ambu.pt")
cap = cv2.VideoCapture(r"C:\Users\Om sinha\Downloads\istockphoto-936189570-640_adpp_is.mp4")
out = cv2.VideoWriter('ambu.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15,(768,432))

bbox = sv.BoundingBoxAnnotator()
lab = sv.LabelAnnotator()

ret = 1

color_index = 0
colors = [(0, 0, 255), (255, 0, 0)]  # Red and Blue in BGR format

# Frame counter to switch colors
frame_count = 0
switch_interval = 3

while ret:
    ret,frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % switch_interval == 0:
        color_index = (color_index + 1) % len(colors)

        # Get current color (alternating red and blue)
    current_color = colors[color_index]

    result = model(frame)[0]

    detect = sv.Detections.from_ultralytics(result)
    l = detect.xyxy
    bboxes_int = [[int(x) for x in bbox] for bbox in l]

    labels = [
        f"#{result.names[class_id]}"
        for class_id in detect.class_id
    ]
    for bbox in bboxes_int:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), current_color, 2)
        cv2.putText(frame,'Ambulance',(x_min,y_min),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)

    out.write(frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()






