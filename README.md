# leukemia-cell-classification
This project focuses on classifying blood cell types using microscopy images. The dataset consists of approximately 12,500 augmented images of white blood cells, labeled into four categories: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. Each class contains roughly 3,000 images.
In addition to the augmented dataset, a smaller original dataset of 410 images is provided with bounding box annotations and subtype labels.
The goal of this project is to train a deep learning model to accurately classify cell types from image data. As an extension, the model may also be used to distinguish between normal and leukemia-related cell patterns.
Different model architectures are explored and compared for performance. While one approach uses a ConvNeXt-based model, this implementation evaluates an alternative architecture to assess differences in accuracy and generalization.

## Run (single-label 4-class, ResNet)

This implementation trains a **ResNet** (not DenseNet) on the original `dataset-master` images by reading `dataset-master/labels.csv` and **filtering to single-label rows** in:
- `EOSINOPHIL`
- `LYMPHOCYTE`
- `MONOCYTE`
- `NEUTROPHIL`

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Train (example):
```bash
python main.py --data-root ../../dataset-2master --model resnet18 --pretrained --epochs 5
```

Notes:
- If you are on Apple Silicon, the script will use **MPS** automatically when available.
- The dataset is imbalanced (especially `MONOCYTE` / `LYMPHOCYTE`), so look at per-class accuracy in addition to overall accuracy.
