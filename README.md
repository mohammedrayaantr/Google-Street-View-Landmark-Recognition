# Google Street View Landmark Recognition

## 1. Project Overview

This project focuses on developing a Landmark Recognition System using Deep Learning and Computer Vision techniques.

The system uses Transfer Learning with a pretrained MobileNetV2 Convolutional Neural Network (CNN) to extract visual features from images and classify them into their corresponding landmark categories.

The model is trained to recognize 10 selected landmark categories using a subset of the Google Landmarks Dataset V2 (GLDv2).

## 2. Project Objectives

- Load and explore a landmark image dataset.
- Preprocess images for deep learning.
- Apply Transfer Learning using MobileNetV2.
- Build and train a landmark classification model.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Generate predictions for unseen landmark images.
- Visualize training performance and classification results.

## 3. Dataset Description

**Dataset:** Google Landmarks Dataset V2 (GLDv2) – Top 51 Categories subset

**Dataset Source:** [Hugging Face – GLDv2 Top 51 Categories](https://huggingface.co/datasets/pemujo/GLDv2_Top_51_Categories)

Dataset details:

- Selected landmark categories: 10
- Total images collected: 300
- Images per category: 30
- Image dimensions: 224 × 224 pixels
- Image channels: RGB

The dataset is accessed through Hugging Face and streamed from a Parquet file.

**Note:** The project uses a subset of GLDv2. The images have not been independently verified as exclusively Google Street View imagery.

## 4. Technologies and Libraries Used

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

This project uses Transfer Learning with MobileNetV2 pretrained on the ImageNet dataset.

The original classification head is removed, and custom classification layers are added.

**Architecture:**

1. MobileNetV2 – Pretrained feature extraction.
2. GlobalAveragePooling2D – Reduces extracted feature maps.
3. Dense Layer – 128 neurons with ReLU activation.
4. Dropout – 0.3 dropout rate to help reduce overfitting.
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

The training dataset is divided into training and validation subsets using an 80:20 validation split.

## 8. Model Evaluation

The trained model is evaluated using the separate test dataset.

**Reported test performance:**

- Test Accuracy: Approximately 83.33%
- Test Images: 60
- Correct Predictions: 50 out of 60

The following evaluation techniques are used:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

## 9. Visualizations

The project generates the following visualizations:

- Sample landmark image visualization.
- Training and validation accuracy graph.
- Training and validation loss graph.
- Confusion matrix heatmap.
- Individual landmark prediction with actual and predicted labels.

## 10. Project Structure

```text
Google_Street_View_Landmark_Recognition/
│
├── Google_Street_View_Landmark_Recognition.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── landmark_recognition.keras
│   └── landmark_classes.json
│
├── results/
│   ├── confusion_matrix.png
│   ├── landmark_prediction.png
│   └── training_validation_metrics.png
│
└── dataset/
```

## 11. Installation and Execution

**Step 1: Clone the repository**

```bash
git clone https://github.com/mohammedrayaantr/Google-Street-View-Landmark-Recognition.git
```

**Step 2: Navigate to the project directory**

```bash
cd Google-Street-View-Landmark-Recognition
```

**Step 3: Install the required libraries**

```bash
pip install -r requirements.txt
```

**Step 4: Launch Jupyter Notebook**

```bash
jupyter notebook
```

**Step 5:** Open `Google_Street_View_Landmark_Recognition.ipynb` and execute the notebook cells in order.

An active internet connection is required to stream the dataset from Hugging Face.

## 12. Saved Model Files

The trained model and its class labels are saved in the `models/` directory.

- `landmark_recognition.keras` – Contains the trained neural network architecture and weights.
- `landmark_classes.json` – Contains the selected landmark category names.

The visual evaluation results are saved in the `results/` directory.

## 13. Limitations

- The project uses a relatively small dataset of 300 images.
- Only 10 selected landmark categories are supported.
- The model shows signs of overfitting, with training accuracy higher than validation accuracy.
- The dataset is not verified to contain exclusively Google Street View images.
- Prediction confidence does not guarantee that a prediction is correct.

## 14. Conclusion

This project demonstrates the application of Deep Learning and Transfer Learning for landmark image classification.

By using a pretrained MobileNetV2 model, the system learns visual features and classifies images into 10 selected landmark categories.

The model achieved approximately 83.33% test accuracy on the selected test dataset.

This project provided practical experience in image preprocessing, transfer learning, model training, evaluation, and computer vision.

## 15. Future Improvements

- Increase the number of landmark images.
- Include more landmark categories.
- Apply data augmentation to improve generalization.
- Experiment with fine-tuning pretrained layers.
- Improve model performance on unseen images.
- Develop a web-based interface for landmark recognition.
