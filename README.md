#  Brain Tumor 3D Segmentation using Deep Learning (BraTS)

A modular, fully tested and production-ready deep learning framework for automatic brain tumor segmentation from multi-modal MRI volumes using a **3D U-Net** architecture.

The project is built around the **BraTS (Brain Tumor Segmentation Challenge)** dataset and has been designed following modern software engineering principles, including modular architecture, comprehensive unit testing, reproducible preprocessing pipelines and reusable training/inference workflows.

Rather than being a simple research prototype, the project aims to provide a maintainable and extensible codebase suitable for experimentation, education and future research.

---

##  Features

-  3D U-Net implementation for volumetric medical image segmentation
-  Native support for the BraTS dataset
-  Fully modular preprocessing pipeline
-  Configurable preprocessing transforms
    - Z-score normalization
    - Isotropic resampling
-  Training-only data augmentation
    - Random flips
    - Random 90° rotations
    - Random Gaussian noise
    - Random gamma correction
    - Random intensity shifting
-  PyTorch Dataset and DataLoader integration
-  Dice + Cross Entropy hybrid loss
-  GPU/CPU compatible training
-  Automatic checkpoint saving
-  Early stopping support
-  Modular inference pipeline
-  Extensive automated test suite (500+ passing tests)
-  Factory builders for reproducible component construction
-  Clean architecture with clear separation of responsibilities

---

## Project Goals

This repository was developed with three primary objectives:

1. Build a complete medical image segmentation framework following modern software engineering practices.

2. Provide a reusable and extensible codebase for future experimentation with novel architectures, preprocessing techniques and loss functions.

3. Demonstrate professional Python development skills through clean architecture, testing and documentation.

---

## Main Technologies

| Category | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Deep Learning | PyTorch |
| Medical Imaging | NiBabel |
| Numerical Computing | NumPy |
| Testing | PyTest |
| MRI Dataset | BraTS |
| Model | 3D U-Net |

---

## Repository Highlights

✔ Modular architecture

✔ Strong separation of concerns

✔ Extensive documentation

✔ Comprehensive automated tests

✔ Production-style project organization

✔ Easily extensible for future research

---
#  Project Architecture

The project follows a modular architecture where each component has a single responsibility. Data loading, preprocessing, model definition, training and inference are completely decoupled, making the framework easy to extend and maintain.

```
                           BraTS Dataset
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   BraTSReader   │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ BraTSDataset    │
                        └─────────────────┘
                                 │
                                 ▼
                 ┌────────────────────────────────┐
                 │   PreprocessingPipeline        │
                 ├────────────────────────────────┤
                 │ • ZScoreNormalization          │
                 │ • ResamplingTransform          │
                 │ • Data Augmentation (train)    │
                 └────────────────────────────────┘
                                 │
                                 ▼
                          PyTorch DataLoader
                                 │
                                 ▼
                          ┌──────────────┐
                          │    UNet3D    │
                          └──────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
      DiceCrossEntropyLoss                 Predictor
                │                                 │
                ▼                                 ▼
            Trainer                      Segmentation Mask
                │
                ▼
        Checkpoints (.pt)
```

The project is intentionally organized so that each module performs one well-defined task:

- **Reader** loads MRI volumes from disk.
- **Dataset** converts patient data into PyTorch tensors.
- **Preprocessing Pipeline** applies deterministic preprocessing.
- **Augmentation** performs stochastic training-only transformations.
- **Model** defines the neural network.
- **Trainer** handles optimization and checkpointing.
- **Predictor** performs inference on unseen data.

No module contains responsibilities belonging to another module, resulting in a clean and maintainable architecture.

---

#  Project Structure

```
project/
│
├── data/
│   ├── raw/
│   │   └── TrainingData/
│   └── processed/
│
├── checkpoints/
│
├── predictions/
│
├── src/
│   │
│   ├── analysis/
│   ├── builders/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   └── visualization/
│
├── tests/
├── scripts/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## Module Overview

| Module | Responsibility |
|---------|----------------|
| **analysis** | Dataset statistics and exploratory analysis |
| **builders** | Factory functions used to construct project components |
| **data** | BraTS reader, dataset implementation and dataloaders |
| **inference** | Prediction pipeline and inference utilities |
| **models** | 3D U-Net architecture, losses and trainer |
| **preprocessing** | Validation, preprocessing and augmentation transforms |
| **utils** | Configuration, checkpoints, early stopping and utilities |
| **visualization** | Visualization helpers for MRI volumes and predictions |
| **tests** | Unit and integration tests |

#  Installation

## Requirements

- Python **3.11** or newer
- Git
- NVIDIA GPU (recommended for training)
- CUDA-compatible version of PyTorch (optional but highly recommended)

The project has been developed and tested using:

| Component | Version |
|----------|---------|
| Python | 3.11 |
| PyTorch | 2.x |
| NumPy | Latest |
| NiBabel | Latest |
| PyTest | Latest |

---

## Clone the repository

```bash
git clone https://github.com/bielvicens/Brain-Tumor-3D-Segmentation.git

cd Brain-Tumor-3D-Segmentation
```

---

## Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

#  Download the BraTS Dataset

This project uses the **Brain Tumor Segmentation Challenge (BraTS)** dataset.

Download the dataset from the official challenge website:

https://www.synapse.org/Synapse:syn25829067

After downloading and extracting the dataset, organize it as follows:

```text
project/

├── data/
│   └── raw/
│       └── TrainingData/
│           ├── BraTS-GLI-00000-000/
│           │      ├── BraTS-GLI-00000-000-t1n.nii.gz
│           │      ├── BraTS-GLI-00000-000-t1c.nii.gz
│           │      ├── BraTS-GLI-00000-000-t2w.nii.gz
│           │      ├── BraTS-GLI-00000-000-t2f.nii.gz
│           │      └── BraTS-GLI-00000-000-seg.nii.gz
│           │
│           ├── BraTS-GLI-00001-000/
│           ├── BraTS-GLI-00002-000/
│           └── ...
```

The project automatically discovers all patient folders.

No manual indexing is required.

---

#  Configuration

The project is configured through the `ProjectConfig` class.

Example configuration:

```python
config = ProjectConfig()

config.training.batch_size = 2
config.training.epochs = 100
config.training.learning_rate = 1e-4

config.model.base_channels = 16

config.training.device = "cuda"
```

Most users will not need to modify the source code.

Training, validation and inference are entirely controlled through the configuration object.

---

#  Training

Launch a complete training session with:

```bash
python train.py
```

During training the framework automatically:

- builds the preprocessing pipeline
- creates the datasets
- performs the train/validation split
- creates DataLoaders
- builds the model
- initializes the optimizer
- computes the Dice + Cross Entropy loss
- saves checkpoints
- applies Early Stopping (if enabled)

Typical workflow:

```
BraTS Dataset
      │
      ▼
Reader
      │
      ▼
Preprocessing
      │
      ▼
Augmentation
      │
      ▼
DataLoader
      │
      ▼
UNet3D
      │
      ▼
Loss
      │
      ▼
Optimizer
      │
      ▼
Checkpoint
```

---

## Training Outputs

After training, checkpoints are automatically saved inside:

```text
checkpoints/
```

Typical output:

```text
checkpoints/

├── best.pt
└── last.pt
```

where:

- **best.pt** stores the model with the lowest validation loss.
- **last.pt** stores the final epoch.

#  Preprocessing Pipeline

Medical image preprocessing is a critical step in deep learning workflows. Rather than embedding preprocessing logic inside the dataset or training code, this project implements a fully modular preprocessing framework based on composable transforms.

Each preprocessing operation inherits from a common `Transform` interface and is executed by a `PreprocessingPipeline`, allowing preprocessing steps to be easily added, removed or reordered without modifying any other component.

This design follows the **Single Responsibility Principle (SRP)** and keeps the data loading, preprocessing and training logic completely independent.

---

## Pipeline Overview

```
Raw MRI Volumes
        │
        ▼
Validation
        │
        ▼
Z-Score Normalization
        │
        ▼
Resampling
        │
        ▼
Training Only?
        │
   ┌────┴─────┐
   │          │
  Yes         No
   │          │
   ▼          ▼
Data       Skip
Augmentation
   │
   ▼
Preprocessed Sample
```

---

## PreprocessingSample

All preprocessing operations work on a single immutable object:

```python
PreprocessingSample
```

This object contains everything required to process a patient:

- Patient identifier
- MRI modalities
- Segmentation mask
- Voxel spacing
- Affine matrix
- Metadata

Instead of modifying the sample in-place, every transform returns a **new** sample using:

```python
sample.replace(...)
```

This guarantees that preprocessing remains deterministic, side-effect free and easy to test.

---

## Transform Interface

Every preprocessing operation inherits from the same abstract base class:

```python
class Transform
```

Each transform implements a single method:

```python
apply(sample)
```

The pipeline automatically performs:

- optional input validation
- transform execution
- exception wrapping
- logging

without requiring every transform to duplicate this logic.

---

## Implemented Preprocessing Transforms

### ZScoreNormalization

Performs independent Z-score normalization for each MRI modality.

For every volume:

\[
x'=\frac{x-\mu}{\sigma}
\]

where

- μ = mean intensity
- σ = standard deviation

This removes scanner-specific intensity scaling while preserving tissue contrast.

---

### ResamplingTransform

Resamples all MRI modalities and segmentation masks to a common voxel spacing.

Benefits:

- isotropic voxel size
- consistent spatial resolution
- identical tensor dimensions across patients

MRI modalities are interpolated using continuous interpolation, while segmentation masks use nearest-neighbor interpolation to preserve discrete labels.

---

#  Data Augmentation

During training the framework automatically performs online data augmentation.

These transformations improve model generalization by exposing the network to plausible variations of the training data.

Importantly, **geometric augmentations are always applied consistently to every MRI modality and the segmentation mask**, preserving voxel correspondence.

Intensity augmentations only modify MRI volumes.

---

## RandomFlip

Randomly flips the complete volume along one spatial axis.

Possible axes:

- Sagittal
- Coronal
- Axial

The same flip is applied to every modality and the segmentation.

---

## RandomRotation90

Applies a random rotation of

- 90°
- 180°
- 270°

around a randomly selected anatomical plane.

Rotations preserve voxel spacing and introduce no interpolation artifacts.

---

## RandomGaussianNoise

Adds zero-mean Gaussian noise to each MRI modality.

The segmentation mask is left unchanged.

This simulates acquisition noise commonly observed in MRI scanners.

---

## RandomGamma

Applies random gamma correction:

\[
I' = I^\gamma
\]

This changes image contrast while preserving anatomical structures.

Gamma values are sampled randomly within a configurable interval.

---

## RandomIntensityShift

Adds a small random intensity offset to every MRI modality.

This simulates scanner-dependent intensity calibration differences.

The segmentation mask is never modified.

---

# Training vs Inference

The framework automatically builds different preprocessing pipelines depending on the application.

## Training

```
Validation
      │
      ▼
Normalization
      │
      ▼
Resampling
      │
      ▼
Random Flip
      │
      ▼
Random Rotation
      │
      ▼
Gaussian Noise
      │
      ▼
Gamma
      │
      ▼
Intensity Shift
```

---

## Validation / Inference

```
Validation
      │
      ▼
Normalization
      │
      ▼
Resampling
```

No stochastic augmentation is applied during validation or inference, ensuring deterministic predictions.

---

# Why This Design?

The preprocessing framework was designed with extensibility in mind.

Adding a new preprocessing operation only requires implementing:

```python
class MyTransform(Transform):

    def apply(self, sample):
        ...
```

No changes are required to:

- Dataset
- DataLoader
- Trainer
- Predictor
- Model
- Existing transforms

The new transform can simply be inserted into the pipeline.

This architecture keeps the preprocessing framework scalable, maintainable and easy to extend for future research.

#  Model Architecture

The segmentation model is based on a **3D U-Net**, one of the most widely adopted architectures for volumetric medical image segmentation.

Unlike conventional 2D convolutional networks, a 3D U-Net processes complete volumetric information using 3D convolutions, allowing the network to exploit contextual information across all three spatial dimensions.

The implementation is intentionally modular, with each building block encapsulated in its own class.

---

## Network Overview

```
                         Input MRI
                    (4 × D × H × W)
                            │
                            ▼
                     DoubleConv3D
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
             Down3D                 Skip Connection
                │                       │
                ▼                       │
             Down3D                 Skip Connection
                │                       │
                ▼                       │
             Down3D                 Skip Connection
                │                       │
                ▼                       │
                  Bottleneck
                DoubleConv3D
                │
                ▼
             Up3D ◄──────────── Skip
                │
                ▼
             Up3D ◄──────────── Skip
                │
                ▼
             Up3D ◄──────────── Skip
                │
                ▼
             Up3D ◄──────────── Skip
                │
                ▼
             1×1×1 Conv3D
                │
                ▼
      Raw Segmentation Logits
```

---

## Encoder

The encoder progressively extracts higher-level semantic features while reducing spatial resolution.

Each encoder stage consists of:

- Max Pooling (downsampling)
- Double 3D Convolution
- Batch Normalization
- ReLU activation

As spatial resolution decreases, the number of feature channels increases.

Example:

```
4 channels
      │
      ▼
16
      ▼
32
      ▼
64
      ▼
128
      ▼
256
```

---

## Bottleneck

The bottleneck represents the deepest level of the network.

Here the receptive field is largest, enabling the model to combine global anatomical context with local tumor appearance.

---

## Decoder

The decoder progressively reconstructs the segmentation mask.

Each decoder stage performs:

1. Transposed convolution (upsampling)
2. Skip connection concatenation
3. DoubleConv3D

Skip connections recover fine spatial information lost during downsampling while preserving high-level semantic features extracted by the encoder.

---

## Skip Connections

Skip connections directly connect encoder features to decoder features of the same spatial resolution.

```
Encoder ─────────────────────────► Decoder
```

They provide:

- sharper tumor boundaries
- improved localization
- faster convergence
- better gradient flow

Without skip connections, much of the fine anatomical detail would be lost.

---

## Output Layer

The final layer is a:

```
Conv3D(kernel_size=1)
```

which projects the decoder features into the desired number of segmentation classes.

For BraTS:

| Class | Meaning |
|--------|---------|
| 0 | Background |
| 1 | Necrotic / Non-enhancing Tumor |
| 2 | Edema |
| 3 | Enhancing Tumor |

The network outputs **raw logits**, allowing the loss function to apply the appropriate activation internally.

---

#  Loss Function

The project uses a hybrid loss that combines:

- Dice Loss
- Cross Entropy Loss

```
Total Loss = Dice Loss + Cross Entropy Loss
```

This combination leverages the strengths of both losses.

---

## Dice Loss

Dice Loss directly optimizes segmentation overlap.

The Dice coefficient is defined as

```
Dice = (2 × |Prediction ∩ GroundTruth|)
       /
       (|Prediction| + |GroundTruth|)
```

Advantages:

- robust to class imbalance
- directly optimizes segmentation quality
- widely used in medical imaging

---

## Cross Entropy Loss

Cross Entropy performs voxel-wise multi-class classification.

It provides stable gradients during the early stages of training and complements Dice Loss.

---

## Why Combine Them?

Dice Loss alone can become unstable during early optimization.

Cross Entropy alone may produce poor overlap on highly imbalanced segmentation tasks.

The hybrid objective combines the strengths of both approaches:

- stable optimization
- improved convergence
- better boundary delineation
- higher segmentation accuracy

This combination is widely adopted in state-of-the-art medical image segmentation models.

---

#  Training Pipeline

Training is handled by the `Trainer` class.

Responsibilities include:

- moving data to the correct device
- forward propagation
- loss computation
- backpropagation
- optimizer update
- validation
- checkpoint saving
- learning-rate scheduler support
- early stopping support

The trainer intentionally contains **no preprocessing logic**, ensuring complete separation between data preparation and optimization.

Training workflow:

```
Dataset
    │
    ▼
DataLoader
    │
    ▼
Forward Pass
    │
    ▼
Loss
    │
    ▼
Backward
    │
    ▼
Optimizer
    │
    ▼
Validation
    │
    ▼
Checkpoint
```

---

## Automatic Checkpoints

During training the framework automatically stores:

```
last.pt
```

The most recent checkpoint.

and

```
best.pt
```

The checkpoint with the lowest validation loss.

Each checkpoint contains:

- model parameters
- optimizer state
- epoch number
- training history

allowing training to be resumed or the best-performing model to be used for inference.

---

## GPU Support

The framework automatically detects CUDA when available.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The same code therefore runs on:

- CPU
- NVIDIA GPU
- Google Colab
- local workstation

#  Inference

Once a model has been trained, inference can be performed using the `predict.py` entry point.

The inference pipeline is intentionally independent from the training pipeline and performs only deterministic preprocessing.

Pipeline:

```
BraTS Patient
      │
      ▼
BraTSReader
      │
      ▼
Preprocessing Pipeline
      │
      ▼
UNet3D
      │
      ▼
Predicted Segmentation
      │
      ▼
prediction.npy
```

Run inference with:

```bash
python predict.py
```

The framework automatically:

- loads the trained checkpoint
- builds the inference preprocessing pipeline
- loads the selected patient
- preprocesses the MRI volumes
- performs segmentation
- saves the predicted mask

Predictions are stored inside:

```text
predictions/
```

Example:

```text
predictions/
└── BraTS-GLI-00001_prediction.npy
```

---

#  Testing

A major objective of this project was software reliability.

The project includes an extensive automated test suite covering every major component of the framework.

Current status:

```
500 passing tests
```

The test suite includes:

- Unit tests
- Integration tests
- Dataset validation
- Preprocessing validation
- Data augmentation
- Model components
- Loss functions
- Trainer
- Predictor
- Builders
- Complete end-to-end training pipeline

Example:

```bash
pytest
```

or

```bash
pytest -v
```

Integration tests verify that the complete workflow executes successfully:

```
BraTS Dataset
      │
      ▼
Dataset
      │
      ▼
Pipeline
      │
      ▼
DataLoader
      │
      ▼
UNet3D
      │
      ▼
Loss
      │
      ▼
Backward
      │
      ▼
Optimizer
```

This ensures every major subsystem works correctly together.

---

#  Results

The framework has been fully implemented and validated through automated testing.

Training experiments and quantitative evaluation will be added after running full-scale experiments on Google Colab using GPU acceleration.

Future versions of this section will include:

- Dice Score
- Validation Loss
- Learning Curves
- Example Segmentations
- Qualitative Comparisons
- Performance Analysis

Example layout:

```
Results/

├── learning_curve.png
├── dice_curve.png
├── prediction_01.png
├── prediction_02.png
└── prediction_03.png
```

---

#  Future Work

The modular architecture allows straightforward extension of the framework.

Possible future improvements include:

- Mixed Precision Training (AMP)
- Sliding Window Inference
- Patch-based Training
- Deep Supervision
- Attention U-Net
- Residual U-Net
- MONAI integration
- Automatic Hyperparameter Search
- Test-Time Augmentation
- ONNX export
- TensorBoard logging
- Weights & Biases integration
- Multi-GPU training
- Distributed Data Parallel
- Additional preprocessing transforms

---

#  References

The implementation was inspired by modern medical image segmentation literature.

- Ronneberger O., Fischer P., Brox T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*

- Çiçek Ö. et al. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.*

- Isensee F. et al. *nnU-Net: A Self-configuring Method for Deep Learning-Based Biomedical Image Segmentation.*

- BraTS Challenge:
  https://www.med.upenn.edu/cbica/brats/

- PyTorch Documentation:
  https://pytorch.org/

- NiBabel Documentation:
  https://nipy.org/nibabel/

---

#  Contributing

Contributions, bug reports and suggestions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Add appropriate tests.
5. Submit a pull request.

All contributions should maintain the project's coding style and include corresponding unit tests whenever applicable.

---

#  License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

#  Author

**Biel Vicens**

Biomedical Engineering Student

Interested in:

- Artificial Intelligence
- Deep Learning
- Medical Imaging
- Computer Vision
- Biomedical Signal Processing
- Medical Image Segmentation

GitHub:

https://github.com/bielvicens

---

#  Acknowledgements

This work was developed as a personal deep learning and software engineering project focused on medical image segmentation.

Special thanks to:

- The BraTS Challenge organizers
- The PyTorch community
- The open-source scientific Python ecosystem