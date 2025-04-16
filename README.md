**Project Overview**

This project implements a deep learning-based system to classify chest X-ray images into four categories: **COVID-19, Normal, Pneumonia**, and **Tuberculosis**. It leverages PyTorch for model training and evaluation, with advanced image preprocessing techniques and visualization tools.

**Key Features**

- Preprocessing Methods: Includes CLAHE, denoising, Butterworth filtering, adaptive masking, histogram equalization, and Gaussian blur.
- Model Architectures: Supports pre-trained models (ResNet50, MobileNetV2, DenseNet121, EfficientNetB0) and a custom CNN.
- Class Imbalance Handling: Uses weighted sampling and class weights.
- Visualization: Plots training history, confusion matrices, ROC curves, classification reports, model predictions, and Grad-CAM heatmaps.
- Experiments: Compares preprocessing methods, model architectures, and balancing techniques.

**Requirements**

To run this project, install the required Python packages:
torch>=2.0.0 torchvision>=0.15.0 numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0 scikit-image>=0.18.0 seaborn>=0.11.0 tqdm>=4.62.0 opencv-python>=4.5.0 scikit-learn>=1.0.0 scipy>=1.7.0 pillow>=8.0.0

**Dataset**

- Structure: Expects a directory chest_xray_data/ with subfolders train/, val/, and test/, each containing subdirectories for the four classes (COVID19, NORMAL, PNEUMONIA, TUBERCULOSIS).

- Source: Chest X-Ray (Pneumonia,Covid-19,Tuberculosis)
   https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis

**Results**

The project trains models with various configurations and evaluates them on a test set. Sample outputs include:

- Training History: Loss and accuracy plots over epochs.
- Confusion Matrix: Visualizes classification performance.
- ROC Curves: Shows model discrimination ability per class.
- Grad-CAM: Highlights regions of interest in images.

Example results (after training):
Test Accuracy: ~95% (best model with parameter).
Macro AUC: ~0.99 (varies by configuration).

The model also tested with different dataset:
1. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
3. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Performance Report
**Overall Accuracy:** 93%  
**Test Samples:** 1,026 images  

| Class          | Precision | Recall | F1-Score | Support Samples |
|----------------|:---------:|:------:|:--------:|:---------------:|
| **COVID19**    | 1.00      | 0.83   | 0.90     | 206             |
| **NORMAL**     | 0.96      | 0.88   | 0.92     | 234             |
| **PNEUMONIA**  | 0.93      | 0.98   | 0.96     | 390             |
| **TUBERCULOSIS** | 0.85    | 1.00   | 0.92     | 196             |

**Macro Average**  
Precision: 0.94 | Recall: 0.92 | F1-Score: 0.92  

**Weighted Average**  
Precision: 0.94 | Recall: 0.93 | F1-Score: 0.93  

**License**

This project is licensed under the MIT License.

