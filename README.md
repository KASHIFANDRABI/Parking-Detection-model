# YOLOv5 Custom Training Guide

This guide provides step-by-step instructions for training a YOLOv5 object detection model on a custom dataset using Google Colab or a local environment. For full implementation beyond training, refer to the YouTube video tutorial: [YOLOv5 Training and Inference](https://www.youtube.com/watch?v=mRhQmRm_egc).

## 1. Clone the YOLOv5 Repository

First, clone the YOLOv5 repository from Ultralytics:

```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!git reset --hard 064365d8683fd002e9ad789c1e91fa3d021b44f0  # Use a stable commit version
```

## 2. Install Dependencies

Ensure all required dependencies are installed:

```bash
!pip install -qr requirements.txt  # Install required packages
```

Import necessary libraries:

```python
import torch
from IPython.display import Image, clear_output
from utils.downloads import attempt_download  # Download models/datasets

# Check setup
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```

## 3. Download and Prepare the Dataset

Use Roboflow to fetch and prepare the dataset:

```bash
!pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("yollolabel").project("hard-hat-sample-hacx2")
version = project.version(2)
dataset = version.download("yolov5")
```

Verify dataset configuration:

```bash
%cat {dataset.location}/data.yaml
```

Extract the number of classes:

```python
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
```

## 4. Train the YOLOv5 Model

Train the model for 100 epochs:

```bash
%cd /content/yolov5/
!python train.py --img 640 --batch 20 --epochs 100 --data {dataset.location}/data.yaml --cfg ./models/yolov5s.yaml --weights '' --name yolov5s_results --cache
```

## 5. Evaluate Model Performance

Plot and display training results:

```python
from utils.plots import plot_results  # Plot results
Image(filename='/content/yolov5/runs/train/yolov5s_results/results.png', width=1000)
```

List trained models:

```bash
%ls runs/train/yolov5s_results/weights
```

## 6. Model Storage and Downloading Weights

After training, all model results, including logs, metrics, and weights, are stored in:

```bash
runs/train/yolov5s_results/
```

The most important file generated is `best.pt`, which contains the optimized model weights. This file is stored in:

```bash
runs/train/yolov5s_results/weights/best.pt
```

Before proceeding further, **ensure that you download the **``** folder** from the above directory. This will allow you to use the trained model for inference.

## 7. Further Implementation (Using OpenCV)

For inference using OpenCV and running detections on new images or videos, refer to the detailed steps in the YouTube tutorial: [YOLOv5 Training and Inference](https://www.youtube.com/watch?v=mRhQmRm_egc). The resources in this GitHub repository match the video tutorial for ease of use.

## Conclusion

This guide covers training YOLOv5 on a custom dataset. Before moving forward, ensure you download the `weights` folder. For using the trained model with OpenCV, refer to the video tutorial for further instructions.

