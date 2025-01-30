# Parking Slot Detection using YOLO

## Overview
This repository contains a Parking Slot Detection model using the YOLO (You Only Look Once) object detection framework. The model is designed to identify and classify parking slots in images or video streams, helping with automated parking management systems.

## Repository Link
[GitHub Repository](https://github.com/KASHIFANDRABI/Parking-Detection-model)

## Features
- Real-time parking slot detection using YOLO
- High accuracy and efficiency
- Supports video and image inputs
- Can be integrated with smart parking systems

## Tools & Dependencies
Ensure you have the following installed:
- Python 3.x
- OpenCV
- YOLOv5 (Ultralytics)
- TensorFlow/Keras (optional, if needed for additional processing)
- NumPy
- Matplotlib (for visualization)
- CUDA (for GPU acceleration, optional but recommended)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KASHIFANDRABI/Parking-Detection-model.git
   cd Parking-Detection-model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download YOLO model weights (if not included in the repo):
   ```bash
   wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
   ```

## Procedure
1. **Prepare the Dataset:**
   - Ensure images/videos contain parking lots.
   - Annotate parking slots if training a custom model.

2. **Load the YOLO Model:**
   - Use a pre-trained YOLO model or train a custom model.
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')
   ```

3. **Run Inference:**
   - Detect parking slots in an image or video.
   ```python
   results = model('test_image.jpg')
   results.show()
   ```

4. **Visualize Results:**
   - Use OpenCV or Matplotlib to display detected parking slots.

5. **Integration:**
   - Implement the model into a real-world application or smart parking system.

## Usage
To run detection on an image:
```bash
python detect.py --image test_image.jpg
```
To run detection on a video:
```bash
python detect.py --video test_video.mp4
```

## Future Enhancements
- Improve accuracy with a custom-trained dataset.
- Implement real-time detection on edge devices.
- Deploy as a web or mobile application.

## Contributing
Feel free to open issues or pull requests to improve the model and its functionality.

## License
This project is licensed under the MIT License.

## Contact
For any queries or collaboration, reach out via [GitHub Issues](https://github.com/KASHIFANDRABI/Parking-Detection-model/issues).

