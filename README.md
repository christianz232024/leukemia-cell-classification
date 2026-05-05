# Overview
This project implements an end-to-end deep learning pipeline for automated classification of white blood cells from microscopy images. The model is based on a fully fine-tuned DenseNet-121 architecture with differential learning rates applied to the backbone and classifier head.

The goal is to accurately classify four major white blood cell types:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

This has applications in medical diagnostics, including infection detection and hematological analysis.

# Features:

DenseNet-121 pretrained on ImageNet
Full backbone fine-tuning with differential learning rates
Advanced data augmentation tailored to microscopy images
Class-weighted loss with label smoothing
Early stopping based on validation accuracy
Cosine annealing learning rate scheduler
Gradient clipping for stable training
Detailed evaluation using classification report and confusion matrix

# Dataset:

The dataset is expected to follow the structure:

dataset/
├── TRAIN/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
└── TEST/
    ├── EOSINOPHIL/
    ├── LYMPHOCYTE/
    ├── MONOCYTE/
    └── NEUTROPHIL/

Update the paths in the script:

TRAIN_DIR = "path/to/TRAIN"
TEST_DIR  = "path/to/TEST"

# Instalation requirements:

Python 3.8+
PyTorch
torchvision
scikit-learn
numpy
tqdm

Install dependencies:

pip install torch torchvision scikit-learn numpy tqdm

# Running script:

Run the training script:

python your_script_name.py

The script will:

Split training data into train/validation sets
Train the model with early stopping

Save the best model to:

densenet_improved.pth
Evaluate performance on the test set

# Note: If you are able, run on a GPU. Model takes aproximately 9 hours to train on device. 
