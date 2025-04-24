# Video Analytics using Deep Learning for Massive Crowd Instant Segments and Detections
# Overview
This project aims to develop a deep learning-based solution for massive crowd segmentation and detection in video analytics. By leveraging advanced neural networks and cutting-edge computer vision techniques, the system can process and analyze crowd scenes in real-time, providing instant segmentations and detections.
# Key Features
Crowd Segmentation and Detection: Detect and segment crowds in videos using deep learning techniques.

Real-time Analysis: The system is designed to handle massive video feeds for real-time crowd analysis.

Deep Learning: Utilizes state-of-the-art models for accurate segmentation and detection in dynamic environments.

# Technologies Used
Deep Learning: Convolutional Neural Networks (CNN), Mask R-CNN, and other advanced models for object detection and segmentation.

Python: The project is developed in Python, utilizing the deep learning ecosystem.

CVAT: CVAT (Computer Vision Annotation Tool) is used for annotating video frames for training the model.

PyCharm 3.7: The development environment used for code editing and running the project.

## Installation

To run this project, make sure to have Python 3.7 installed, along with the necessary libraries. Here are the steps to set up the environment:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/repository-name.git
cd repository-name
```

### 2. Install Dependencies
Create a virtual environment and install dependencies using pip:
```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```
## Dataset
For this project, a total of 1596 images were extracted from a video. These images will be used for the crowd segmentation and detection tasks.


## Annotation
Use [CVAT](https://github.com/openvinotoolkit/cvat) to annotate the dataset for training. Follow the CVAT documentation for setting up and annotating your video frames. To annotate the dataset using CVAT for training purposes, and the images can be used for deep learning model training.

## Running the Project
When the dataset is ready, run the following command to start the video analytics pipeline:
```bash
python main.py
```
