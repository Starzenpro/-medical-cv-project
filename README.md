# Medical CV Project: Pneumonia Detection from Chest X-Rays

A PyTorch implementation for binary classification of chest X-ray images, distinguishing between normal lungs and pneumonia.

## Overview

This repository demonstrates a complete workflow for training, validating, and evaluating a convolutional neural network on the Chest X-Ray Images (Pneumonia) dataset. The project includes data loading utilities, a compact CNN architecture, training/evaluation scripts, and automated result logging.

## Dataset

The dataset is organized into separate splits under `data/chest_xray/`:
- Training: 5,216 images
- Validation: 16 images
- Test: 624 images

Images are labeled in two categories:
- `NORMAL`
- `PNEUMONIA`

The data loader resizes images to `224x224`, converts them to RGB, and applies normalization consistent with standard pretrained vision models.

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- tqdm

Install required packages with:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Starzenpro/-medical-cv-project.git
cd -medical-cv-project
```

2. Extract the dataset archive if needed, or ensure the dataset is already available at:

```bash
data/chest_xray/
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training and Evaluation

Run the main training script from the repository root:

```bash
cd src
python train.py
```

The training script automatically:
- selects `cuda` if available, otherwise uses `cpu`
- builds data loaders for `train`, `val`, and `test`
- trains the model for 5 epochs
- saves the best validation model to `best_model.pth`
- generates a training history plot at `training_history.png`
- evaluates the final model on the test set

## Output Artifacts

- `best_model.pth`: Saved state of the model with the best validation accuracy
- `training_history.png`: Loss and accuracy curves for training and validation
- Console output includes a classification report and confusion matrix for the test set

## Project Structure

- `src/model.py`: Defines the `PneumoniaCNN` architecture
- `src/train.py`: Training loop, validation routine, testing, and plotting
- `src/utils.py`: Custom `ChestXrayDataset` and data loader creation
- `data/`: Dataset directory with `train`, `val`, and `test` splits
- `notebooks/`: Notebook exploration and experimentation files
- `requirements.txt`: Python library dependencies

## Model Architecture

The CNN architecture consists of:
- 3 convolutional blocks with `Conv2d` + `ReLU` + `MaxPool2d`
- 1 fully connected layer with 512 units
- dropout regularization (0.5)
- final linear layer for binary classification

The network is trained using:
- loss: `CrossEntropyLoss`
- optimizer: `Adam`
- batch size: 32

## Notes

- The current validation split is small (16 images), so consider using cross-validation or a larger validation set for more robust model selection.
- The model is intentionally compact for quick experimentation on limited hardware.
- Future improvements may include transfer learning, data augmentation, and more advanced architectures such as ResNet or EfficientNet.

## License

This project is provided for educational and demonstration purposes.