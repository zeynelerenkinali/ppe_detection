# PPE Detection with YOLOv8

A lightweight PPE (Personal Protective Equipment) detection system using YOLOv8. Detects safety gear (hardhats, masks, vests) and violations in real-time. Trained for 50 epochs on a custom dataset.

---

## Features
- Real-time detection (video/webcam).
- Flags PPE violations (e.g., no mask).
- Custom confidence thresholding.
- Pre-trained weights included.

---

## Quick Start

**Installation**  
- Clone the repository:  
  `git clone https://github.com/zeynelerenkinali/ppe-detection-yolov8.git`  
- Install dependencies:  
  `pip install ultralytics opencv-python cvzone torch`  
- Place `ppe.pt` (pre-trained weights) in the project root.

**Usage**  
- For video detection:  
  `python ppe_detection.py --video videos/ppe-3.mp4`
- For webcam detection:  
  `python ppe_detection.py --webcam`

---

## Train Your Model  
- Prepare dataset in YOLO format (images + labels).  
- Update `data.yaml` with your dataset paths and class names.  
- Run training in Google Colab:  
  ```python
  from ultralytics import YOLO
  model = YOLO("yolov8l.pt")
  model.train(data="data.yaml", epochs=50, imgsz=640)
---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgments  
- [Ultralytics YOLOv8](https://ultralytics.com)  
- Dataset contributors.  

---
