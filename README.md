# DSA-HGN: Dynamic Sparse Adaptive Hypergraph Network for Skeleton-Based Action Recognition

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Note**: This project is currently in the experimental stage. Results and implementations are subject to change.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Zoo](#model-zoo)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

DSA-HGN is a novel deep learning framework designed for skeleton-based action recognition tasks. It introduces a **Dynamic Sparse Adaptive Hypergraph Network** that leverages hypergraph convolutions with entropy-regularized soft sparsity to model complex joint relationships in human skeletal data.

### Supported Datasets

- **EgoGesture**: Egocentric hand gesture recognition (83 classes, 21 joints)
- **SHREC'17 Track**: 3D hand gesture recognition (14 classes, 22 joints)

### Multi-Stream Architecture

The framework supports four complementary data streams:
- **Joint Stream**: Raw joint coordinates
- **Bone Stream**: Bone vectors between connected joints
- **Joint-Motion Stream**: Temporal differences of joints
- **Bone-Motion Stream**: Temporal differences of bones

## ✨ Key Features

### 1. Dynamic Sparse Hypergraph Module
- **Entropy-Regularized Softmax**: Replaces hard Top-K selection with differentiable soft sparsity
- **Learnable Prototypes**: Orthogonally initialized hyperedge prototypes
- **Gradient Flow**: Ensures all prototypes receive gradients during training

### 2. Dual-Branch Architecture
- **Spatial-Temporal Branch**: Captures standard ST-GCN patterns
- **Channel-Differential Branch**: Models inter-channel relationships

### 3. Hypergraph Attention Fusion Module (HAFM)
- Adaptive weighting of multiple streams
- End-to-end learnable fusion strategy

### 4. Hardware Compatibility
- Native support for **Apple Silicon (MPS)** backend
- CUDA and CPU fallback support
- Mixed-precision training ready

## 🏗️ Architecture

```
Input Skeleton Sequence (N, C, T, V, M)
         ↓
    Data BN Layer
         ↓
┌────────────────────────┐
│   10-Layer ST-GCN      │
│   with Hypergraph      │
│   Convolution          │
└────────────────────────┘
         ↓
  Global Average Pool
         ↓
    Dropout Layer
         ↓
   FC Classifier (num_classes)
```

### Hypergraph Convolution Unit

```
Node Features (N, C, T, V)
         ↓
    Query Projection
         ↓
  Prototype Matching
         ↓
Entropy-Regularized Softmax → Incidence Matrix H (N, V, M)
         ↓
  V2E Aggregation (H @ X)
         ↓
  Edge Convolution
         ↓
  E2V Propagation (H^T @ E)
         ↓
   Residual + BN + ReLU
```

## 🔧 Installation

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.10.0
CUDA >= 11.1 (optional, for GPU training)
```

### Setup Environment

```bash
# Clone repository
git clone https://github.com/yourusername/DSA-HGN.git
cd DSA-HGN

# Create conda environment
conda create -n dsa_hgn python=3.8
conda activate dsa_hgn

# Install PyTorch (example for CUDA 11.3)
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia

# For Apple Silicon (M1/M2/M3)
# PyTorch with MPS support is automatically available in recent versions

# Install dependencies
pip install -r requirements.txt

# Install torchlight module
cd torchlight
python setup.py install
cd ..
```

### Requirements

```txt
numpy>=1.19.0
pyyaml>=5.4.0
tensorboardX>=2.4.0
h5py>=3.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
networkx>=2.5.0
tqdm>=4.60.0
```

## 📊 Dataset Preparation

### SHREC'17 Track Dataset

1. **Download Dataset**:
   ```bash
   # Download from official source
   # http://www-rech.telecom-lille.fr/shrec2017-hand/
   ```

2. **Data Structure**:
   ```
   DATA/
   └── SHREC2017_data/
       ├── train_data.npy      # Shape: (N_train, C, T, V, M)
       ├── train_label.pkl     # List: [sample_names, labels]
       ├── val_data.npy        # Shape: (N_val, C, T, V, M)
       └── val_label.pkl
   ```

3. **Update Config Paths**:
   ```yaml
   # config/SHREC/joint/joint.yaml
   train_feeder_args:
     data_path: /path/to/DATA/SHREC2017_data/train_data.npy
     label_path: /path/to/DATA/SHREC2017_data/train_label.pkl
   
   test_feeder_args:
     data_path: /path/to/DATA/SHREC2017_data/val_data.npy
     label_path: /path/to/DATA/SHREC2017_data/val_label.pkl
   ```

### EgoGesture Dataset

1. **Download and Extract**:
   ```bash
   # Follow CTR-GCN preprocessing pipeline
   # https://github.com/Uason-Chen/CTR-GCN
   ```

2. **Data Structure**:
   ```
   data/
   └── egogesture/
       ├── train_data.npy
       ├── train_label.pkl
       ├── val_data.npy
       └── val_label.pkl
   ```

### Data Format

**NumPy Array Format** (`.npy`):
```python
Shape: (N, C, T, V, M)
# N: Number of samples
# C: Number of channels (typically 3 for x, y, z)
# T: Temporal length (number of frames)
# V: Number of joints (21 for EgoGesture, 22 for SHREC)
# M: Number of persons (typically 1 for hand gestures)
```

**Label Format** (`.pkl`):
```python
[sample_names, labels]
# sample_names: List of string identifiers
# labels: List of integer class labels
```

## 🚀 Training

### Single Stream Training

#### Joint Stream

```bash
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --work_dir work_dir/SHREC/hyperhand_joint \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

#### Bone Stream

```bash
python main.py finetune_evaluation \
    --config config/SHREC/bone/bone.yaml \
    --work_dir work_dir/SHREC/hyperhand_bone \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

#### Joint-Motion Stream

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Jmotion/jmotion.yaml \
    --work_dir work_dir/SHREC/hyperhand_jmotion \
    --device 0
```

#### Bone-Motion Stream

```bash
python main.py finetune_evaluation \
    --config config/SHREC/Bmotion/bmotion.yaml \
    --work_dir work_dir/SHREC/hyperhand_bmotion \
    --device 0
```
#### 连续运行joint和jmotion
```bash
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --work_dir work_dir/SHREC/joint \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60 \
    && \
python main.py finetune_evaluation \
    --config config/SHREC/Jmotion/jmotion.yaml \
    --work_dir work_dir/SHREC/jmotion \
    --device 0 \
    --batch_size 32 \
    --num_epoch 60
```

### Resume Training from Checkpoint

```bash
python main.py finetune_evaluation \
    --config config/SHREC/bone/bone_fi.yaml \
    --weights work_dir/SHREC/hyperhand_bone/epoch025_acc87.14_model.pt \
    --start_epoch 25
```

### Multi-Stream Fusion Training

```bash
python main.py finetune_evaluation \
    --config config/SHREC/fusion/hafm_fusion.yaml \
    --work_dir work_dir/SHREC/hyperhand_hafm_fusion \
    --device 0 \
    --batch_size 16  # Reduced batch size due to memory requirements
```

### Training on Apple Silicon

```bash
# MPS backend is automatically detected
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --device 0 \
    --use_gpu True
```

### Key Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `base_lr` | Initial learning rate | 0.05 | 0.05 (SGD), 0.001 (AdamW) |
| `num_epoch` | Total training epochs | 60 | 60-150 |
| `batch_size` | Batch size per GPU | 32 | 32 (single stream), 16 (fusion) |
| `lambda_entropy` | Entropy regularization weight | 0.001 | 0.001-0.005 |
| `lambda_ortho` | Orthogonality loss weight | 0.1 | 0.1 |
| `grad_clip_norm` | Gradient clipping threshold | 1.0 | 1.0 |
| `num_hyperedges` | Number of hypergraph edges | 16 | 16 |

## 🧪 Evaluation

### Single Model Evaluation

```bash
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --phase test \
    --weights work_dir/SHREC/hyperhand_joint/best_model.pt \
    --save_result True
```

### Multi-Stream Ensemble Evaluation

#### Three-Stream Ensemble (SHREC)

```bash
python ensemble_shrec.py
```

**Configuration in `ensemble_shrec.py`**:
```python
# Path configuration
joint_path = './work_dir/SHREC/hyperhand_joint/test_result.pkl'
bone_path = './work_dir/SHREC/hyperhand_bone/test_result.pkl'
motion_path = './work_dir/SHREC/hyperhand_motion/test_result.pkl'
label_path = '/path/to/SHREC2017_data/val_label.pkl'

# Fusion weights [Joint, Bone, Motion]
alpha = [1.0, 0.6, 0.8]  # Example weights
```

#### Two-Stream Ensemble (EgoGesture)

```bash
python ensemble_egogesture.py
```

**Configuration in `ensemble_egogesture.py`**:
```python
# Path configuration
joint_path = 'work_dir/egogesture/aimclr_finetune_joint/test_result.pkl'
bone_path = 'work_dir/egogesture/aimclr_finetune_bone/test_result.pkl'
label_path = '/path/to/egogesture/val_label.pkl'

# Fusion weights [Joint, Bone]
alpha = [0.5, 0.5]  # Equal weighting
```

### Performance Analysis Tools

#### Confusion Matrix Visualization

```bash
python tools/Confusion\ Matrix.py
```

Generates `SHREC_Confusion_Matrix.png` showing per-class prediction distribution.

#### Error Analysis

```bash
python tools/Error\ Analysis.py
```

Outputs:
- Top-5 confusion pairs
- Per-class error rates
- Misclassification patterns

#### Topology Visualization

```bash
python tools/visualize_topology.py \
    work_dir/SHREC/hyperhand_joint/topology_best_epoch_50.npy \
    --threshold 0.1
```

Visualizes learned hypergraph virtual connections.

## 📦 Model Zoo

> **Note**: Models are currently in experimental stage. Pre-trained weights will be released upon paper acceptance.

Expected performance (subject to change):

| Dataset | Stream | Epochs | Top-1 Acc | Config |
|---------|--------|--------|-----------|--------|
| SHREC'17 | Joint | 60 | ~85% | `config/SHREC/joint/joint.yaml` |
| SHREC'17 | Bone | 60 | ~87% | `config/SHREC/bone/bone.yaml` |
| SHREC'17 | J-Motion | 60 | ~82% | `config/SHREC/Jmotion/jmotion.yaml` |
| SHREC'17 | B-Motion | 60 | ~84% | `config/SHREC/Bmotion/bmotion.yaml` |
| SHREC'17 | Ensemble (3-stream) | - | ~95% | - |
| EgoGesture | Joint | 60 | TBD | `config/egogesture/supervised/hyperhand_supervised.yaml` |

## 📁 Project Structure

```
DSA-HGN/
├── config/                          # Configuration files
│   ├── SHREC/
│   │   ├── joint/                   # Joint stream configs
│   │   ├── bone/                    # Bone stream configs
│   │   ├── Jmotion/                 # Joint-motion configs
│   │   ├── Bmotion/                 # Bone-motion configs
│   │   └── fusion/                  # Multi-stream fusion configs
│   └── egogesture/
│       └── supervised/
├── feeder/                          # Data loading and augmentation
│   ├── feeder_egogesture.py        # Main data feeder
│   └── tools.py                     # Augmentation functions
├── graph/                           # Graph topology definitions
│   ├── shrec.py                     # SHREC'17 skeleton graph
│   ├── egogesture.py               # EgoGesture skeleton graph
│   └── tools.py                     # Graph utilities
├── net/                             # Network architectures
│   ├── dsa_hgn.py                  # Main DSA-HGN model
│   ├── hypergraph_modules.py       # Hypergraph convolution layers
│   ├── basic_modules.py            # GCN and TCN modules
│   └── utils/                       # Network utilities
├── processor/                       # Training and evaluation logic
│   ├── processor.py                # Base processor class
│   ├── recognition.py              # Recognition processor
│   └── io.py                        # I/O operations
├── tools/                           # Analysis and visualization
│   ├── Confusion Matrix.py         # Confusion matrix generation
│   ├── Error Analysis.py           # Error pattern analysis
│   └── visualize_topology.py       # Topology visualization
├── torchlight/                      # Training utilities
│   └── io.py                        # Model I/O and logging
├── ensemble_shrec.py               # SHREC ensemble evaluation
├── ensemble_egogesture.py          # EgoGesture ensemble evaluation
├── main.py                          # Main entry point
└── README.md                        # This file
```

## 🔬 Key Configuration Parameters

### Model Architecture

```yaml
model_args:
  in_channels: 3                     # Input channels (x, y, z)
  base_channels: 64                  # Base feature dimension
  num_stages: 10                     # Number of ST-GCN layers
  inflate_stages: [5, 8]            # Layers to double channels
  down_stages: [5, 8]               # Layers to apply temporal downsampling
  num_hyperedges: 16                # Number of hypergraph prototypes
  adaptive: true                     # Enable adaptive graph learning
  use_virtual_conn: True            # Enable hypergraph connections
  drop_out: 0.0                      # Dropout rate (0 for training stability)
```

### Data Augmentation

```yaml
train_feeder_args:
  window_size: 180                   # Temporal window length
  normalization: False               # Apply z-score normalization
  random_choose: True               # Random temporal cropping
  random_shift: True                # Random temporal shifting
  random_rot: True                  # Random rotation augmentation
  shear_amplitude: 0.5              # Shear transformation strength
  temperal_padding_ratio: 6         # Temporal padding ratio
  repeat: 5                          # Dataset repetition factor
```

### Training Strategy

```yaml
optimizer: SGD                       # Optimizer (SGD/Adam/AdamW)
base_lr: 0.05                        # Initial learning rate
weight_decay: 0.0005                # L2 regularization
nesterov: True                       # Use Nesterov momentum
grad_clip_norm: 1.0                 # Gradient clipping threshold

# Learning rate schedule
step: [30, 50]                      # LR decay milestones
lr_decay_rate: 0.1                  # LR decay factor
warm_up_epoch: 5                    # Warmup epochs

# Regularization
lambda_entropy: 0.001               # Soft sparsity weight
lambda_ortho: 0.1                   # Prototype orthogonality weight
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size
   --batch_size 16
   # Or use gradient accumulation (modify processor.py)
   ```

2. **MPS Backend Issues (Mac)**:
   ```bash
   # If MPS fails, force CPU mode
   --use_gpu False
   ```

3. **Data Loading Errors**:
   ```python
   # Check data paths in config files
   # Ensure .npy and .pkl files exist
   # Verify data shape: (N, C, T, V, M)
   ```

4. **NaN Loss During Training**:
   ```yaml
   # Reduce learning rate
   base_lr: 0.01  # Instead of 0.05
   
   # Enable gradient clipping
   grad_clip_norm: 1.0
   
   # Increase entropy weight
   lambda_entropy: 0.005
   ```

### Debug Mode

```bash
# Use reduced dataset for quick testing
python main.py finetune_evaluation \
    --config config/SHREC/joint/joint.yaml \
    --debug True \
    --num_epoch 2
```

## 📝 Configuration Examples

### Minimal Working Example

```yaml
# config/SHREC/minimal.yaml
work_dir: ./work_dir/SHREC/minimal_test

model: net.dsa_hgn.Model
model_args:
  in_channels: 3
  num_class: 14
  num_point: 22
  base_channels: 64
  num_hyperedges: 16
  graph: graph.shrec.Graph

train_feeder: feeder.feeder_egogesture.Feeder
train_feeder_args:
  data_path: /path/to/train_data.npy
  label_path: /path/to/train_label.pkl
  bone: False
  vel: False
  window_size: 180

optimizer: SGD
base_lr: 0.05
num_epoch: 10
batch_size: 32
device: [0]
```

Run with:
```bash
python main.py finetune_evaluation --config config/SHREC/minimal.yaml
```

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir work_dir/SHREC/hyperhand_joint

# View at http://localhost:6006
```

**Available Metrics**:
- Training loss (CE + Entropy + Orthogonality)
- Learning rate schedule
- Validation accuracy (Top-1, Top-5)
- Per-epoch statistics

### Log Files

```bash
# View training log
tail -f work_dir/SHREC/hyperhand_joint/log.txt

# Check saved models
ls work_dir/SHREC/hyperhand_joint/*.pt
```

## 🔍 Hyperparameter Tuning

### Learning Rate Search

```bash
# Test different learning rates
for lr in 0.01 0.05 0.1; do
    python main.py finetune_evaluation \
        --config config/SHREC/joint/joint.yaml \
        --base_lr $lr \
        --work_dir work_dir/SHREC/lr_${lr}
done
```

### Entropy Weight Search

```bash
# Test different entropy weights
for lambda_e in 0.0001 0.001 0.005 0.01; do
    python main.py finetune_evaluation \
        --config config/SHREC/joint/joint.yaml \
        --lambda_entropy $lambda_e \
        --work_dir work_dir/SHREC/entropy_${lambda_e}
done
```

## 🤝 Contributing

This project is currently in experimental stage. Contributions are welcome after the initial release.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests (when available)
pytest tests/

# Code formatting
black . --line-length 120
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **CTR-GCN**: Base architecture inspiration [[GitHub](https://github.com/Uason-Chen/CTR-GCN)]
- **SHREC'17 Track**: Hand gesture dataset [[Website](http://www-rech.telecom-lille.fr/shrec2017-hand/)]
- **EgoGesture**: Egocentric gesture dataset [[Paper](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html)]

## 📮 Contact

For questions and feedback:
- Open an issue on GitHub
- Email: [your-email@example.com] (to be updated)

---

**Note**: This README reflects the current experimental state of the project. Performance numbers, model architectures, and implementation details may change as development progresses.
