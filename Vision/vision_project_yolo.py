from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')

video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()
    if ret:
        result = model.track(frame, persist=True)
        frame_ = result[0].plot()
        cv2.imshow('frame' , frame_)
        if cv2.waitKey(25) and 0xff == ord('q'):
            break