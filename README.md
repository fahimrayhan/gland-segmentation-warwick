# Biomedical Image Segmentation — Warwick QU Dataset

PyTorch implementation of three segmentation models (SimpleSegNet, UNetSkip, ResidualUNet) for bone surface segmentation in ultrasound images using the Warwick QU Dataset.

## Dataset

Warwick QU Dataset with 128×128 RGB ultrasound images and binary masks.

- Train: 68 samples (80/20 train/val split)
- Test: 60 samples

## Models

- **SimpleSegNet** — baseline encoder–decoder without skip connections
- **UNetSkip** — U-Net with skip connections (concatenation)
- **ResidualUNet** — U-Net with residual blocks, BatchNorm, and dropout

## Optimization Techniques

### Loss & Optimizer
- **BCEWithLogitsLoss** — numerically stable (sigmoid + BCE fused) instead of Sigmoid + BCELoss
- **Adam optimizer** with learning rate `1e-3` and **weight decay** for L2 regularization
- **Dice score with smoothing (`1e-6`)** to avoid division by zero

### Learning Rate Scheduling
- **`ReduceLROnPlateau` scheduler** — adaptively reduces LR when validation Dice plateaus
- **Validation-based stepping** — scheduler steps on **val Dice** (not test) to avoid leakage

### Regularization
- **Weight decay** on the Adam optimizer to penalize large weights
- **Dropout2d (p=0.1)** in the ResidualUNet bottleneck
- **BatchNorm** in residual blocks for stable convergence
- **Early stopping** via best-val-Dice tracking

### Data Augmentation (train only)
- Random **horizontal flip** (p=0.5)
- Random **vertical flip** (p=0.5)
- Random **rotation** (±15°)
- **Color jitter** (brightness=0.2, contrast=0.2)

### Architectural Optimizations
- **Skip connections** (concatenation) to preserve spatial detail
- **Residual connections** (shortcut addition) to improve gradient flow

### Reproducibility & Efficiency
- Deterministic training — `torch.manual_seed(42)`, `np.random.seed(42)`, cuDNN benchmark disabled
- `torch.inference_mode()` during validation for memory/speed
- Proper train/val split to prevent data leakage

## Training Config

| Setting | Value |
|---|---|
| Batch size | 8 |
| Learning rate | 1e-3 |
| Epochs | 100 |
| Loss | BCEWithLogitsLoss |
| Metric | Sørensen–Dice |
| Scheduler | ReduceLROnPlateau (on val Dice) |

## Results

| Metric | Value |
|---|---|
| Final Train Loss | 0.4222 |
| Best Val Dice | 84.52% (Epoch 91) |
| Final Val Dice | 83.27% |

## Usage

```bash
git clone https://github.com/<your-username>/biomedical-image-segmentation-warwick.git
cd biomedical-image-segmentation-warwick
pip install torch torchvision numpy pandas matplotlib pillow tqdm
jupyter notebook biomedical-image-segmentation-warwick.ipynb
```

#### Adjust the hyperparameters at the top of the notebook: 
```code
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 100
data_root = "/path/to/WARWICK"
```
### Tech Stack
PyTorch · TorchVision · NumPy · Pandas · Matplotlib · PIL · tqdm

### Acknowledgements
Warwick QU Dataset
U-Net (Ronneberger et al.) and ResNet (He et al.) architectures
