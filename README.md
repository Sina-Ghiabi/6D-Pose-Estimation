# 🎯 6D Pose Estimation with EnhancedRCVPose + YOLOv8

A full pipeline for **6D object pose estimation** on the **LineMOD dataset**, combining YOLOv8 object detection with an EnhancedRCVPose model that uses RGB-D input, Feature Pyramid Networks, attention modules, and radius map supervision.

---

## 📁 Project Structure

```
6D-pose-estimation-main/
│
├── 01_setup.ipynb          # Environment setup, Drive mount, dataset download
├── 02_preprocess.ipynb     # Pose extraction, radius maps, train/val/test splits
├── 03_yolo_train.ipynb     # YOLOv8s object detection training
├── 04_pose_train.ipynb     # EnhancedRCVPose training (2-stage + fine-tune)
├── 05_evaluate.ipynb       # Validation + test evaluation (Trans RMSE, Rot, ADD)
├── 06_visualize.ipynb      # Pose wireframe, YOLO detections, radius map heatmaps
│
├── src/
│   ├── model.py            # EnhancedRCVPose architecture + WeightedPoseLoss
│   └── dataset.py          # PoseDataset, safe_collate, augmentation
│
├── configs/
│   └── linemod_final.yaml  # Dataset configuration
│
├── checkpoints/
│   └── yolo_model.pt       # Example YOLO checkpoint
│
├── sample_output/          # Sample prediction images
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Google Colab)

Run the notebooks **in order**:

| Step | Notebook | Action |
|------|----------|--------|
| 1 | `01_setup.ipynb` | Mount Drive, install packages, clone repo, extract LineMOD |
| 2 | `02_preprocess.ipynb` | Extract poses, compute radius maps, build splits |
| 3 | `03_yolo_train.ipynb` | Train YOLOv8s for bounding box detection |
| 4 | `04_pose_train.ipynb` | Train EnhancedRCVPose (stage 1 + 2 + fine-tune) |
| 5 | `05_evaluate.ipynb` | Compute validation and test metrics |
| 6 | `06_visualize.ipynb` | Visualize predictions with 3D mesh overlays |

> Each notebook reads from `/content/config.json` written by `01_setup.ipynb` — no hardcoded paths.

---

## ⚠️ Disk Space Strategy: Process Classes in Batches of 3

Extracting and preprocessing the full 13-class LineMOD dataset **all at once does not fit**
on a standard Google Colab local disk (~113 GB) — this is especially true once
`02_preprocess.ipynb` generates 9 full-resolution radius maps per frame. Trying to extract
everything in one pass reliably ends in `OSError: [Errno 28] No space left on device`,
even when the extraction target is Google Drive itself (Colab's Drive mount still stages
every write through local disk before syncing, so it hits the same wall).

The workflow that actually works — **process 3 classes at a time**:

1. Pick **3 classes** (edit `EXTRACT_THESE_CLASSES` in `01_setup.ipynb` Cell 6).
2. Run `01_setup.ipynb` → `02_preprocess.ipynb` for just those 3 classes — extraction through
   all preprocessing steps (poses, radius maps, splits), stopping **before** modeling.
3. Save the preprocessed output for those 3 classes back to Google Drive so it persists.
4. **Fully disconnect the Colab runtime** (`Runtime → Disconnect and delete runtime`) to
   reclaim local disk — a plain "Restart runtime" does *not* free disk, only the kernel.
5. Reconnect, pick the **next 3 classes**, and repeat steps 1–4.
6. Once all 13 classes have been extracted + preprocessed this way and saved to Drive, move
   on to `03_yolo_train.ipynb` / `04_pose_train.ipynb` (modeling) — training needs all classes
   present together (it uses `ConcatDataset` across classes, so partial/sequential training
   per class is **not** an option — that causes catastrophic forgetting).

---

## 🏗️ Model Architecture: EnhancedRCVPose

```
RGB  (3, H, W)  ──► ResNet50 backbone ──► FPN ──► Attention ──┐
                                                                ├──► Fusion (512→256 conv)
Depth (1, H, W) ──► ResNet50 backbone ──► FPN ──► Attention ──┘
                                                      │
                                          ┌───────────┴───────────┐
                                   Global AvgPool           Outside9 Head
                                          │                        │
                                    Pose Head                (9, H, W) radius maps
                                          │
                                  [tx, ty, tz, qx, qy, qz, qw]
```

**Key components:**
- **Dual ResNet50 backbone** — separate encoders for RGB and depth
- **Feature Pyramid Network (FPN)** — multi-scale feature extraction
- **Self-Attention modules** — spatial attention on fused features
- **Pose head** — outputs 7-D pose vector `[translation (3) + quaternion (4)]`
- **Outside9 head** — outputs 9 radius maps `(H, W)` for keypoint supervision

---

## 📊 Training Strategy

### Stage 1 — Warm-up (epochs 0–14, backbone frozen)
- Only pose head and outside9 head are trained
- OneCycleLR scheduler with `max_lr=1e-3`
- Prevents destroying pretrained ResNet50 features

### Stage 2 — Full fine-tune (epochs 15–79, backbone unfrozen)
- All parameters trained with lower LR (`3e-4`)
- CosineAnnealingLR scheduler
- Early stopping with `patience=15`

### Stage 3 — Rotation fine-tune (10 epochs)
- Backbone frozen again
- `W_ROT=15` (higher weight on geodesic rotation loss)
- Typically gives **1–2° improvement** in rotation error

### GPU Optimisations
| Technique | Benefit |
|-----------|---------|
| `torch.compile()` | +15–25% throughput (PyTorch ≥ 2.0) |
| AMP (`autocast` + `GradScaler`) | ~2× faster, FP16 activations |
| Gradient accumulation (×4) | Effective batch = 32 |
| `pin_memory` + `persistent_workers` | Faster CPU→GPU transfers |
| `prefetch_factor=2` | Reduces data loading bottleneck |

---

## 📉 Loss Function: WeightedPoseLoss

```
L = W_TRANS × L_trans + W_ROT × L_rot + W_PTS × L_pts
```

| Component | Formula | Default Weight |
|-----------|---------|---------------|
| `L_trans` | L1 loss on translation `[tx, ty, tz]` | 1.0 |
| `L_rot` | Geodesic loss: `arccos(\|q_pred · q_gt\|)` | 10.0 |
| `L_pts` | MSE on 9 radius maps `(H, W)` | 1.0 |

Rotation is weighted 10× higher because angular error is harder to minimise.

---

## 📐 Evaluation Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Translation RMSE** | √ mean(‖t_pred − t_gt‖²) | < 2 cm |
| **Rotation Error** | 2·arccos(\|q̂·q̂_gt\|) | < 5° |
| **Points MSE** | mean((R_pred − R_gt)²) | lower is better |
| **ADD** | mean‖(R_p·m + t_p) − (R_g·m + t_g)‖ | < per-class threshold |
| **ADD Success %** | frames with ADD < threshold | > 90% |

---

## 🗃️ Dataset: LineMOD

**13 objects** with known 3D models (APE, BENCHVISE, BOWL, CAMERA, CAN, CAT, CUP, DRILLER, DUCK, EGGBOX, GLUE, HOLEPUNCHER, IRON, LAMP, PHONE).

**Data split per class:**
- 80% → train
- 10% → val
- 10% → test (held-out, evaluate only once!)

**Per-sample data:**
```
data/XX/
  rgb/          ← RGB images (.png)
  depth/        ← Depth images (.dpt, uint16 mm)
  pose/         ← Ground-truth pose matrices (.npy, 4×4)
  mask/         ← Object masks (.png)
  Out_pt1_dm/   ← Radius map for keypoint 1 (.npy)
  ...
  Out_pt9_dm/   ← Radius map for keypoint 9 (.npy)
  Split/        ← train.txt / val.txt / test.txt
  mesh.ply      ← 3D mesh for ADD metric + visualization
  gt.yml        ← Bounding boxes for YOLO training
```

---

## 📦 Requirements

```
torch >= 2.0
torchvision
ultralytics       # YOLOv8
open3d            # Mesh loading for ADD metric + visualization
scipy             # Quaternion utilities
opencv-python
Pillow
numpy
pandas
matplotlib
tqdm
PyYAML
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

All paths and settings are centralised in `/content/config.json` (created by `01_setup.ipynb`):

```json
{
  "DATA_DIR": "/content/dataset/linemod/.../data",
  "YOLO_DIR": "/content/Linemod_ready",
  "DRIVE_MODELS": "/content/drive/MyDrive/models",
  "REPO_DIR": "/content/6D-pose-estimation-main",
  "ALL_CLASSES": ["01","02","04","05","06","08","09","10","11","12","13","14","15"],
  "CLASS_NAMES": ["ape","benchvise","bowl","camera","can","cat","cup","driller","duck","eggbox","glue","holepuncher","iron"],
  "CAMERA_K": [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]],
  "ADD_THRESHOLDS": {"01": 0.01421, "05": 0.02841, "08": 0.03187, ...},
  "YOLO_YAML": "/content/linemod_yolo.yaml",
  "YOLO_MODEL_PATH": "...Drive.../yolo_best.pt",
  "BEST_POSE_MODEL": "...Drive.../best_rcvpose_YYYYMMDD_HHMMSS_finetuned.pth"
}
```

---

## 🖼️ Sample Outputs

| Original | GT Pose (red) | Predicted Pose (green) |
|----------|--------------|----------------------|
| ![](sample_output/pose_estimate_pred1.jpg) | — | ![](sample_output/pose_estimate_pred2.jpg) |

---

## 📚 References

- **RCVPose**: Xu et al., *RCVPose: Recovery of 3D Pose from Radial Correspondences*
- **YOLOv8**: Ultralytics — [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **LineMOD Dataset**: Hinterstoisser et al., *Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes*
- **ADD Metric**: Xiang et al., *PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes*
