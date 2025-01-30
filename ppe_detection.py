from ultralytics import YOLO
import torch
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) # for webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("videos/ppe-3.mp4") # for video
if not cap.isOpened():
    print("Error: Could not open video file.")


model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

#Check is cuda available
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            # w, h = x2-x1,y2-y1
            # bbox = int(x1),int(y1),int(w),int(h)
            # cvzone.cornerRect(img,bbox)

            # Confidence
            conf = math.ceil(box.conf[0]*100)/100
            
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == 'Mask':
                    varColor = (0, 255 ,0)
                elif currentClass == 'Person' or currentClass == 'machinery' or currentClass == 'vehicle' or currentClass == 'Safety Cone':
                    varColor = (0, 255, 255)
                else:
                    varColor = (0, 0, 255)
                cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(13,x1+13),max(30,y1-15)), scale=1, thickness=2, offset=5, colorB=varColor, colorR=varColor)
                cv2.rectangle(img, (x1,y1), (x2,y2), varColor, 3)



    cv2.imshow("Image", img)
    cv2.waitKey(1)

