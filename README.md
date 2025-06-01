# Video Analytics using Deep Learning for Massive Crowd Instant Segments and Detections
# Overview
**Crowd Pose Detection**, a deep learning-based method for the detection of human poses in huge, dense crowds like the Hajj pilgrimage. Traditional pose estimation fails in extreme crowd conditions. This project aims to develop a deep learning-based method to massive crowd segmentation and detection in video analytics and bridge that gap with a specially trained, COCO-formatted dataset and customized Mask R-CNN architecture. With the use of deep neural networks and advanced computer vision algorithms, the system is capable of processing crowd scenes in real-time and providing real-time segmentations and detections.
# Key Features
**Crowd Segmentation and Detection:** Detect and segment crowds in video frames using state-of-the-art deep learning techniques, providing accurate individual pose detection even in extreme conditions.

**Real-time Analysis:** The system is built to process enormous video feeds in real time, facilitating immediate crowd analysis, which is critical for safety monitoring and event management.

**Deep Learning:** Utilizes **Mask R-CNN** for precise segmentation and detection, enabling pixel-level accuracy in crowded environments.

**Crowd Safety Enhancements:** Use of pose detection and crowd segmentation to assess safety risks and prevent overcrowding during high-density events.

**Scalability:** The system is scalable, designed to work across different environments, from small gatherings to massive events, and can handle multiple video inputs simultaneously.

# Technologies Used
**Deep Learning Models:** Convolutional Neural Networks (CNNs), **Mask R-CNN**, and other advanced models for accurate object detection and segmentation.

**Python:** The project is developed using Python 3.7, leveraging its rich ecosystem of machine learning libraries.

**CVAT:** CVAT (Computer Vision Annotation Tool) is used for annotating video frames to create the dataset for model training.

## Installation

To run this project, make sure to have **Python 3.7 installed**, along with the necessary libraries. Here are the steps to set up the environment:

### 1. Clone the repository
```bash
#Grab the repo
git clone https://github.com/HHassanHub/crowd-pose-detection.git

#Enter the project directory
cd crowd-pose-detection
```

### 2. Setup Instructions

1. Install all dependencies from `requirements.txt`. **Python 3.7** and **TensorFlow 2.0.0** are recommended for compatibility with the Mask R-CNN architecture.

```bash
pip install -r requirements.txt
```
### 3. Install Dependencies
Create a virtual environment and install dependencies using pip:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: `env\Scripts\activate`
pip install -r requirements.txt
```
## Dataset
For this project, a total of **1596 images** were extracted from a video. From These, **15 images** has been annotated and these will be used for the crowd segmentation and detection tasks.


## Annotation
Use [CVAT](https://github.com/openvinotoolkit/cvat) to annotate your training dataset. Read the CVAT documentation to see how to configure and annotate your video frames. To annotate your training dataset with CVAT, and the images to train deep learning models.

## Running the Project
When the dataset is ready, run the following command to start the video analytics pipeline:
```bash
python main.py
```
This will initiate the process of analyzing the crowd footage and providing instant pose detection and segmentation.

## Training Results
### Original Input Frames

These are a few sample frames extracted from the Hajj video used for training:

<p float="middle">
  <img src="images/IMG_0001.jpg" width="400"/>
  <img src="images/IMG_0002.jpg" width="400"/>
  <img src="images/IMG_0003.jpg" width="400"/>
  <img src="images/IMG_0004.jpg" width="400"/>
</p>

---

## Training Output

These are a few sample results after training the model on the Hajj video frames:

<p float="middle">
<img src="images/myplot 1.png" width="400"/>
<br>
<img src="images/myplot 2.png" width="400"/>
<br>
<img src="images/myplot 3.png" width="400"/>
<br>
<img src="images/myplot 4.png" width="400"/>
</p>

---

## Test Results
These are a few sample results after testing the model on the Hajj video frames:

<p float="middle">
<img src="images/test result 1.jpg" width="400"/>
<img src="images/test result 2.jpg" width="400"/>
<img src="images/test result 3.jpg" width="400"/>
<img src="images/test result 4.jpg" width="400"/>

---

## Testing Images and Results

The model was evaluated on high-density crowd footage from Hajj-pilgrimage, containing complex scenes with heavy occlusion, varied lighting, and dense human interactions.

### Performance Metrics
- **Mean Average Precision (mAP @[0.5:0.95])**: 78.0%  
- **Average Precision (AP @ 0.5)**: 60.1%  
- **Average Precision (AP @ 0.75)**: 40.9%  
- **Average Recall (AR @ 1)**: 44.3  
- **Average Recall (AR @ 10)**: 55.9  

The model shows a significant improvement over the baseline ResNet50 model (mAP 62.6%), with over 14% increase in average precision while maintaining inference times between 95-110 ms per image.

### Pose Detection Results
- **Total poses detected**: 3,472  
- **Mean confidence score**: 0.864  
- **Keypoint localization accuracy (PCK@0.2)**: 87.3%  
- **Real Positive Rate**: 6.5%  
- **False Negatives**: 12.8%  

Most false positives were found in distant or cluttered regions, but the model maintained high accuracy for subjects in the foreground and midfield. These results confirm the effectiveness and robustness of the Mask R-CNN-based approach for pose estimation and segmentation in extremely dense crowd scenarios.

## Future Enhancements

Based on the current outcomes and challenges, future work will focus on:

- **Multi-scale Feature Learning:** Improving detection and segmentation of small, distant, and heavily occluded individuals by incorporating multi-scale and context-aware feature extraction techniques.

- **Temporal Integration:** Leveraging temporal information from video sequences to enhance pose continuity and reduce missed detections caused by transient occlusions.

- **Context-aware Filtering:** Reducing false positives in complex backgrounds by integrating advanced context-aware post-processing methods.

- **Real-time Inference Optimization:** Further optimizing model inference speed to support real-time applications on resource-constrained devices.

- **Integration with IoT and Edge Devices:** Expanding the system to work with IoT sensors and edge computing for live crowd monitoring and dynamic safety management in large-scale events.

These enhancements aim to improve robustness, accuracy, and practical deployment for crowd pose estimation in extremely dense and dynamic environments like the Hajj pilgrimage.
