# Google Street View Landmark Recognition

A deep learning project that classifies landmark images into 10 categories using Transfer Learning and the MobileNetV2 architecture.

## 1. Project Overview

This project implements a Landmark Recognition System using Deep Learning and Computer Vision.

The system uses a pretrained MobileNetV2 Convolutional Neural Network (CNN) to extract visual features from landmark images and classify them into their corresponding categories.

The model is trained on a subset of the Google Landmarks Dataset V2 (GLDv2), using 10 selected landmark categories.

## 2. Project Objectives

- Explore and understand a landmark image dataset.
- Perform image preprocessing and preparation.
- Apply Transfer Learning using MobileNetV2.
- Build and train a landmark classification model.
- Evaluate the model using classification metrics.
- Predict landmark categories from unseen images.
- Visualize model performance and prediction results.

## 3. Dataset Description

**Dataset:** Google Landmarks Dataset V2 (GLDv2) – Top 51 Categories subset

**Dataset Source:** [Hugging Face – GLDv2 Top 51 Categories](https://huggingface.co/datasets/pemujo/GLDv2_Top_51_Categories)

Dataset details:

- Selected categories: 10
- Total images collected: 300
- Images per category: 30
- Image dimensions after preprocessing: 224 × 224 pixels
- Image format: RGB

The dataset is streamed from a Parquet file hosted on Hugging Face.

**Note:** This project uses a subset of GLDv2. The images have not been independently verified as exclusively Google Street View imagery.

## 4. Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

## 5. Model Architecture

The project uses Transfer Learning with MobileNetV2 pretrained on the ImageNet dataset.

The original classification layer is removed, and a custom classification head is added.

**Model architecture:**

1. MobileNetV2 – Pretrained feature extraction.
2. GlobalAveragePooling2D – Reduces the extracted feature maps.
3. Dense Layer – 128 neurons with ReLU activation.
4. Dropout – 0.3 dropout rate.
5. Output Layer – 10 neurons with Softmax activation.

The pretrained MobileNetV2 layers are frozen during training.

## 6. Image Preprocessing

The following preprocessing steps are performed:

- Convert images into RGB format.
- Resize images to 224 × 224 pixels.
- Convert images into NumPy arrays.
- Apply MobileNetV2 preprocessing using `preprocess_input`.
- Convert landmark categories into numerical labels.

## 7. Model Training

The model is compiled and trained using the following configuration:

| Parameter | Value |
|---|---|
| Pretrained Model | MobileNetV2 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Epochs | 15 |
| Batch Size | 16 |
| Training Images | 240 |
| Validation Images | 48 |
| Testing Images | 60 |

The dataset is divided into training and testing sets using an 80:20 split. An additional 20% of the training data is used for validation during model training.

## 8. Model Evaluation

The trained model is evaluated using a separate test dataset.

**Test Results:**

| Metric | Result |
|---|---|
| Test Accuracy | 83.33% |
| Total Test Images | 60 |
| Correct Predictions | 50 |
| Incorrect Predictions | 10 |

The project uses the following evaluation techniques:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

## 9. Results and Visualizations

The project generates the following visualizations:

- Sample landmark image.
- Training and validation accuracy graph.
- Training and validation loss graph.
- Confusion matrix heatmap.
- Landmark prediction showing actual and predicted labels.

The generated result images are included in this repository.

## 10. Project Structure

```text
Google-Street-View-Landmark-Recognition/
│
├── Google_Street_View_Landmark_Recognition.ipynb
├── README.md
├── requirements.txt
│
├── landmark_recognition.keras
├── landmark_classes.json
│
├── confusion_matrix.png
├── landmark_prediction.png
└── training_validation_metrics.png
```

## 11. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/mohammedrayaantr/Google-Street-View-Landmark-Recognition.git
```

### Step 2: Navigate to the project directory

```bash
cd Google-Street-View-Landmark-Recognition
```

### Step 3: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the notebook

Open `Google_Street_View_Landmark_Recognition.ipynb` and execute the cells in order.

**Note:** An active internet connection is required to access the dataset from Hugging Face. TensorFlow and its dependencies must also be installed in a compatible Python environment.

## 12. Saved Model Files

The trained model and its class labels are included in the repository.

- `landmark_recognition.keras` – Saved trained neural network model, including its architecture and weights.
- `landmark_classes.json` – Contains the landmark category labels used by the model.

The following result images are also included:

- `confusion_matrix.png`
- `landmark_prediction.png`
- `training_validation_metrics.png`

## 13. Project Limitations

- The project uses a relatively small dataset of 300 images.
- Only 10 selected landmark categories
