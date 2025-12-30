# Automatic MedSAM

**Automatic Medical Image Segmentation with Weak Supervision**

This project implements the "Automatic MedSAM" model that adapts the Segment Anything Model (MedSAM version) for automatic medical image segmentation using weak supervision (tight bounding boxes) and a lightweight trainable Prompt Module.

## 🎯 Key Features

- **Weak Supervision**: Train using only bounding box annotations instead of full segmentation masks
- **Automatic Prompt Generation**: Trainable Prompt Module generates prompts automatically
- **Memory Efficient**: Offline embedding strategy + AMP for training on 8GB VRAM GPUs
- **Mathematical Constraints**: Novel loss function with tightness and size constraints

## 📁 Project Structure

```
Ours/
├── config.py           # Configuration dataclass with hyperparameters
├── dataset.py          # Data loading with weak label simulation
├── model.py            # PromptModule and AutoMedSAM architecture
├── loss.py             # Custom loss functions (L_empty, L_tightbox, L_size)
├── visualization.py    # Training progress visualization
├── train.py            # Two-phase training engine
├── main.py             # Entry point
├── utils.py            # Utility functions
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## 🔧 Installation

```bash
# Clone the repository
cd "d:\HCMUS\Year-4\Sem-1\CV\Project\Ours"

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download MedSAM Checkpoint

Download the MedSAM ViT-B checkpoint and place it in the project root:

```bash
# Download from: https://drive.google.com/drive/folders/1bWv_Zs5oYLpGMAvbotnlNXJPq7ltRUvF
# Place as: ./medsam_vit_b.pth
```

## 📊 Data Preparation

Organize your data in the following structure:

```
data/
├── images/
│   ├── sample001.png  (or .nii.gz for NIfTI)
│   ├── sample002.png
│   └── ...
└── masks/
    ├── sample001.png  (binary segmentation masks)
    ├── sample002.png
    └── ...
```

**Supported Formats**:

- 2D Images: PNG, JPEG, TIFF
- 3D Volumes: NIfTI (.nii, .nii.gz)

## 🚀 Training

### Quick Start (Synthetic Data)

Test the pipeline with synthetic data:

```bash
python main.py --synthetic --epochs 10 --batch_size 4
```

### Full Training

```bash
python main.py \
    --data_dir ./data \
    --medsam_checkpoint ./medsam_vit_b.pth \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.001 \
    --lambda_tightness 0.0001 \
    --lambda_size 0.01
```

### Key Arguments

| Argument             | Default  | Description                         |
| -------------------- | -------- | ----------------------------------- |
| `--data_dir`         | `./data` | Path to data directory              |
| `--epochs`           | `100`    | Number of training epochs           |
| `--batch_size`       | `4`      | Batch size (optimized for 8GB VRAM) |
| `--lr`               | `0.001`  | Learning rate                       |
| `--lambda_tightness` | `0.0001` | Tightness constraint weight (λ₁)    |
| `--lambda_size`      | `0.01`   | Size constraint weight (λ₂)         |
| `--no_amp`           | `False`  | Disable Automatic Mixed Precision   |
| `--synthetic`        | `False`  | Use synthetic data for testing      |

## 📐 Architecture

### Prompt Module

The trainable Prompt Module consists of two branches:

1. **Dense Branch**: 1×1 Conv → ReLU → 3×3 Conv → Dense Embedding (256×64×64)
2. **Sparse Branch**: 1×1 Conv → ReLU → MaxPool → FC → Point/Box Coordinates

### Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{empty} + \lambda_1 \cdot \mathcal{L}_{tightbox} + \lambda_2 \cdot \mathcal{L}_{size}$$

- **L_empty**: Forces background outside bounding box to have low probability
- **L_tightbox**: Ensures prediction touches all sides of the bounding box
- **L_size**: Constrains foreground area to be within reasonable bounds

The tightness constraint uses a pseudo log-barrier function:

$$\psi_t(x) = \begin{cases} -\frac{1}{t}\log(-x) & \text{if } x \leq -\frac{1}{t^2} \\ tx - \frac{1}{t}\log(\frac{1}{t^2}) + \frac{1}{t} & \text{otherwise} \end{cases}$$

## 💾 Memory Optimization

For RTX 4060 (8GB VRAM):

1. **Offline Embedding Strategy**: Pre-compute image embeddings in Phase 1
2. **Automatic Mixed Precision (AMP)**: FP16 computation where possible
3. **Frozen MedSAM Backbone**: Only train the lightweight Prompt Module

## 📈 Outputs

Training produces:

```
output/
├── checkpoints/
│   ├── best_model.pth      # Best performing model
│   └── model_epoch_X.pth   # Periodic checkpoints
├── logs/
│   ├── loss_curves.png     # Training curves
│   └── training_history.json
└── debug_snapshots/
    └── snapshot_eXXX_bXXXX.png  # Visual progress
```

### Visual Snapshots

Each snapshot includes:

- Original image with bounding boxes (GT: green, Pred: red)
- Ground truth mask
- Predicted soft mask (heatmap)
- Predicted binary mask
- Metrics overlay (Dice score, loss values)

## 🔬 API Usage

```python
from config import AutoMedSAMConfig
from model import create_model
from loss import AutoMedSAMLoss, compute_dice_score

# Create configuration
config = AutoMedSAMConfig(
    batch_size=4,
    learning_rate=1e-3,
    use_amp=True
)

# Create model
model = create_model(
    medsam_checkpoint_path="./medsam_vit_b.pth",
    device="cuda"
)

# Forward pass with pre-computed embeddings
output = model.forward_with_embeddings(image_embeddings, return_prompts=True)
masks = output['masks']
prompts = output['prompts']

# Compute loss
criterion = AutoMedSAMLoss(lambda_tightness=1e-4, lambda_size=1e-2)
loss_output = criterion(masks, bboxes)

# Get metrics
dice = compute_dice_score(masks, ground_truth_masks)
```

## 📚 References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [MedSAM: Segment Anything in Medical Images](https://github.com/bowang-lab/MedSAM)

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Meta AI for the Segment Anything Model
- MedSAM team for the medical image adaptation
