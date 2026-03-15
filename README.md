# Real-Time ASL Fingerspelling Recognition with ResNet-18

CPSC 599 Project

## Authors

| Name | UCID |
|------|------|
| Jaspinder Singh Maan | 30209953 |
| Ayham Al-wshah | 30262427 |
| Rumeza Fatima | 30244910 |

---

## Abstract

A real-time ASL fingerspelling (alphabet) recognition system designed to run on everyday hardware using a standard webcam. The pipeline captures live frames via OpenCV, localizes the signing hand using MediaPipe Hands, applies a padded crop around the detected hand landmarks, and classifies the cropped region with a ResNet-18 model fine-tuned for the 29-class ASL alphabet task. The system works reliably under typical indoor conditions, with remaining errors primarily occurring under challenging lighting, motion blur, distant hands, and visually similar letters.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Training Procedure](#training-procedure)
- [Training Results](#training-results)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project performs real-time ASL fingerspelling recognition at the alphabet level (A-Z plus SPACE, DELETE, and NOTHING -- 29 classes total). Rather than building a heavy sequence model, the goal is to deliver a practical end-to-end system that runs in real time on common hardware and produces readable, stable predictions.

The system combines a robust hand localization stage with a lightweight image classifier:

1. **Hand Detection** -- MediaPipe Hands localizes the hand in each webcam frame and returns 21 landmark points.
2. **ROI Cropping** -- Landmark coordinates are converted into a padded bounding box and the hand region is cropped, reducing background influence and improving input consistency.
3. **Classification** -- The cropped hand image is preprocessed (resized to 224x224, ImageNet normalization) and passed through a fine-tuned ResNet-18, which outputs the predicted ASL letter and confidence score.
4. **Display** -- The prediction, bounding box, and hand landmarks are overlaid on the live video feed.

---

## System Architecture

```
Webcam Frame
    |
    v
MediaPipe Hands (landmark detection, single hand, confidence >= 0.6)
    |
    v
Bounding Box from Landmarks (with ~35% padding)
    |
    v
Hand ROI Crop
    |
    v
Preprocess (resize 224x224, ImageNet normalization)
    |
    v
ResNet-18 (ImageNet pretrained backbone)
    |
    v
Dropout(0.3) -> Linear(512 -> 29 classes)
    |
    v
Predicted Letter + Confidence Score -> Overlay on Frame
```

**Design rationale:**

- **MediaPipe cropping + ResNet-18** runs fast enough for interactive webcam use on common hardware.
- **ROI cropping** reduces background clutter and improves consistency between training data and webcam frames.
- **MediaPipe** provides stable hand landmarks without requiring training a separate detector.

---

## Project Structure

```
sign-translate/
|-- main.py                  # Real-time webcam inference pipeline
|-- graphs.py                # Parses training logs and generates loss plots
|-- test_mp.py               # Diagnostic script for verifying MediaPipe installation
|-- asl_resnet18_best.pth    # Trained model checkpoint (weights + class list)
|-- requirements.txt         # Python dependencies
|-- loss_full.png            # Full training/validation loss plot
|-- loss_zoomed.png          # Zoomed loss plot
|-- train_loss_zoomed.png    # Training loss (zoomed)
|-- val_loss_zoomed.png      # Validation loss (zoomed)
```

---

## Datasets

**Primary dataset:** ASL Alphabet dataset [2] -- contains labeled images for the ASL fingerspelling alphabet across 29 classes (A-Z, SPACE, DELETE, NOTHING).

**Preprocessing pipeline:**
- All images resized to 224x224 pixels
- Normalized using ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

**Data augmentation during training:**
- Color jitter (brightness, contrast, saturation, hue) applied with 80% probability
- Random affine transformations: rotations up to 10 degrees, translations (8% horizontal/vertical), scaling (0.9x-1.1x)

We originally planned to incorporate video-based data from WLASL [1] to improve robustness, but time and computational constraints prevented full integration.

---

## Training Procedure

- **Backbone:** ResNet-18 initialized with ImageNet pretrained weights
- **Classification head:** Dropout(0.3) followed by Linear layer
- **Loss function:** Cross-entropy
- **Optimizer:** AdamW
- **Environment:** Google Colab / local GPU
- **Regularization:** Dropout, weight decay, and strong data augmentation

---

## Training Results

The model was trained for 9 epochs after correcting a data leakage issue (see below). Loss metrics from the corrected training run:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 0.2466 | 92.60% | 0.0139 | 99.57% |
| 2 | 0.0967 | 97.02% | 0.0085 | 99.73% |
| 3 | 0.0745 | 97.76% | 0.0177 | 99.34% |
| 4 | 0.0609 | 98.14% | 0.0033 | 99.89% |
| 5 | 0.0493 | 98.52% | 0.0033 | 99.87% |
| 6 | 0.0439 | 98.69% | 0.0014 | 99.95% |
| 7 | 0.0400 | 98.77% | 0.0171 | 99.52% |
| 8 | 0.0345 | 98.97% | 0.0121 | 99.58% |
| 9 | 0.0367 | 98.91% | 0.0002 | 99.99% |

Best checkpoint saved at epoch 6 (val accuracy: 99.95%).

**Note on suspiciously high validation accuracy:** The validation accuracy figures above are unusually high and should be interpreted with caution. During development, we noticed that validation accuracy saturated near 100%, which did not match the model's real-time webcam performance. Upon investigation, we discovered that the ASL Alphabet dataset contains sequential video frames -- meaning that a random train/val split causes nearly identical frames to appear in both sets, resulting in data leakage. This inflates validation accuracy and does not reflect true generalization. We relied primarily on loss curves and qualitative webcam testing to assess the model's real-world performance, which is noticeably lower than what these numbers suggest.

---

## Known Issues and Limitations

- **Data leakage in evaluation:** As noted above, the dataset's sequential frames cause artificially inflated validation accuracy under random splitting.
- **Domain gap:** Clean dataset images differ significantly from real webcam inputs (lighting variation, motion blur, background clutter, hand distance).
- **Visually similar letters:** Letters that differ only by subtle finger positions (e.g., M/N, R/U) are frequently confused.
- **Motion blur and distance:** Fast hand movement and distance from the camera degrade classification accuracy.
- **No temporal smoothing in final demo:** A sliding-window prediction buffer was prepared but not enabled in the final release for simplicity.

---

## Requirements

- Python 3.10+
- A webcam
- (Optional) NVIDIA GPU with CUDA for faster inference

Key dependencies:

- PyTorch 2.8+
- torchvision 0.23+
- OpenCV (opencv-python)
- MediaPipe
- Pillow
- NumPy

See `requirements.txt` for the full pinned dependency list.

---

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/sign-translate.git
   cd sign-translate
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   MediaPipe is not listed in `requirements.txt` and must be installed separately:

   ```bash
   pip install mediapipe
   ```

---

## Usage

### Run the real-time recognizer

```bash
python main.py
```

- A window titled **"ASL V2 - MediaPipe Crop"** will open showing your webcam feed.
- Hold up an ASL fingerspelling sign in front of the camera.
- The predicted letter and confidence score appear in the top-left corner.
- Press **q** to quit.

### Generate training loss graphs

```bash
python graphs.py
```

Parses the embedded training log and saves `train_loss_zoomed.png` and `val_loss_zoomed.png`.

### Verify MediaPipe installation

```bash
python test_mp.py
```

Prints the MediaPipe version and confirms that the `solutions.hands` module can be imported.

---

## Future Work

- **Noisy-data fine-tuning:** Integrate letter-only subsets from WLASL video data to expose the model to more realistic conditions and signer variation.
- **Temporal smoothing:** Enable the prepared sliding-window majority vote and confidence gating to reduce prediction flicker.
- **In-domain data collection:** Collect a small webcam-captured dataset to close the train-test domain gap.
- **Proper evaluation split:** Use a subject-aware or video-aware split to eliminate data leakage and obtain reliable accuracy metrics.
- **Extended recognition:** Scale from letter-level to word- and sentence-level recognition.

---

## References

[1] W. N. Hennes, "Sign Language Dataset (WLASL Videos)," Kaggle, 2022. Available: https://www.kaggle.com/datasets/waseemnagahhenes/sign-language-dataset-wlasl-videos

[2] A. Nagaraj, "ASL Alphabet," Kaggle, 2018. Available: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
