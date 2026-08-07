# Face-Recognition-Classifier
# 🎭 Image Face Recognition & Classification System using ANN

An end-to-end Computer Vision and Deep Learning pipeline built with **Python**, **TensorFlow/Keras**, and **OpenCV**. This system preprocesses facial image datasets, trains a multi-layer Artificial Neural Network (ANN) classifier, and provides real-time inference to identify distinct celebrity faces.

---

## 📌 Project Overview

This project implements an Object-Oriented Programming (OOP) workflow to automate face classification across multi-class datasets. It handles image ingestion, feature normalization, neural architecture setup, dynamic model training, model persistence (serialization), and prediction pipeline evaluation.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.x
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV (`cv2`)
- **Data Processing & ML:** NumPy, scikit-learn
- **File System Utilities:** `os`, `glob`

---

## 📐 Model Architecture

| Layer Type | Nodes / Shape | Activation Function |
| :--- | :--- | :--- |
| **Input / Flatten** | $100 \times 100 \times 3 \rightarrow 30,000$ | — |
| **Dense (Hidden 1)** | 128 | ReLU |
| **Dense (Hidden 2)** | 64 | ReLU |
| **Dense (Hidden 3)** | 32 | ReLU |
| **Output Layer** | 10 Classes | Softmax |

* **Optimizer:** Adam
* **Loss Function:** `sparse_categorical_crossentropy`
* **Metrics:** Accuracy

---

## 🚀 Key Features

* **Automated Data Preprocessing:** Loads image folders dynamically, resizes images to uniform dimensions ($100 \times 100$), and normalizes pixel intensities to the $[0, 1]$ range.
* **OOP Architecture:** Encapsulates preprocessing, compilation, training, and inference methods cleanly inside a single class.
* **Model Serialization:** Automatically saves the trained model (`.model` format) for rapid deployment and future re-use without re-training.
* **Inference Pipeline:** Single-function execution (`predectation`) to pass raw unseen images and output human-readable predictions.

---

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/face-recognition-ann.git](https://github.com/your-username/face-recognition-ann.git)
   cd face-recognition-ann
