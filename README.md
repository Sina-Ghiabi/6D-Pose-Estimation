# 🎯 6D Pose Estimation — EnhancedRCVPose + YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sina-Ghiabi/6D-Pose-Estimation/blob/master/01_setup.ipynb)

A complete, Colab-first pipeline for **6D object pose estimation** on the **LINEMOD** dataset:
an RGB-D dual-backbone network (**EnhancedRCVPose**) predicts each object's full 3D rotation +
translation, refined at inference time with depth-based ICP, alongside a **YOLOv8** detector for
2D bounding boxes.

Six notebooks take you from raw dataset zips to a trained, evaluated, visualized model —
`01_setup` → `02_preprocess` → `03_yolo_train` → `04_pose_train` → `05_evaluate` → `06_visualize`.

### Contents
[Results](#-results) · [Architecture](#%EF%B8%8F-architecture--enhancedrcvpose) · [Key Engineering Decisions](#-key-engineering-decisions) · [Quick Start](#-quick-start-google-colab) · [Training & Loss](#-training-strategy--loss) · [Evaluation](#-evaluation-pipeline) · [Dataset](#%EF%B8%8F-dataset-linemod) · [Project Structure](#-project-structure) · [Configuration](#-configuration) · [References](#-references)

---

## 📊 Results

Validation set, 13 LINEMOD classes, held-out split never seen during training:

| Metric | Mean | What it means |
|---|---|---|
| **Translation RMSE** | **~2.2 cm** | 3D position error |
| **Rotation Error** | **~8.5°** | geodesic angular error |
| **ADD Success Rate** | **~98.8%** | fraction of frames with correct pose (ADD metric, per-class threshold) |

Most classes individually land translation error under 2 cm and rotation error under 8°; a
couple of harder/larger objects (e.g. `benchvise`) run higher — see `05_evaluate.ipynb`'s
per-class table for the full breakdown.

---

## 🏗️ Architecture — EnhancedRCVPose

```
RGB  (3, H, W)  ──► ResNet50 backbone ──► FPN ──► Attention ──┐
                                                                ├──► Fusion (concat + conv)
Depth (1, H, W) ──► ResNet50 backbone ──► FPN ──► Attention ──┘
                                                      │
                                          ┌───────────┴───────────┐
                                   Global AvgPool            Outside9 Head
                                          │                        │
                                    Pose Head                (9, H, W) radius maps
                                          │                  (auxiliary supervision)
                        [tx, ty, tz, 6-D continuous rotation]
                                          │
                              decoded to [t, quaternion] for
                              ICP refinement / metrics / viz
```

- **Dual ResNet50 backbone** — separate pretrained encoders for RGB and depth (depth `conv1`
  initialised from the mean of the RGB `conv1` weights).
- **FPN + self-attention** on each modality, fused by concatenation + conv.
- **Pose head → 9-D output**: 3 translation + a **6-D continuous rotation representation**
  ([Zhou et al., 2019](https://arxiv.org/abs/1812.07035)) instead of a raw quaternion. Quaternions
  have a discontinuity/double-cover ambiguity (`q` and `-q` are the same rotation) that makes them
  measurably harder for a network to regress directly; decoding 6-D → a proper rotation matrix via
  Gram-Schmidt avoids that. The loss compares rotations via the geodesic angle between matrices
  (`arccos((trace(Rᵀ·R_gt) − 1) / 2)`) — the same angular quantity the earlier quaternion formula
  used, so the loss weight didn't need retuning when this changed.
- **Outside9 head** — 9 per-pixel radius maps (distance from each pixel to one of 9 FPS-sampled
  3D keypoints), trained as auxiliary supervision.

---

## 🔬 Key engineering decisions

A few points worth knowing before reading the notebooks — these came out of debugging real
accuracy problems, not just architecture choices made up front.

### Radius maps are computed on-the-fly, never stored to disk
Earlier versions of this pipeline precomputed 9 folders per class (`Out_pt1_dm` … `Out_pt9_dm`),
one `.npy` file *per training image per keypoint*. That's a 9× multiplier on top of the already
large LINEMOD dataset — far too much to fit on a Colab session's local disk, let alone transfer
from Drive. `PoseDataset` (`src/dataset.py`) now computes all 9 radius maps **at load time**,
directly from `depth` + `mask` + `pose` + `Outside9.npy` (just 9 keypoint coordinates, not
thousands of files), using a `numba`-JIT-compiled kernel run in parallel across `DataLoader`
workers. Local disk per class now stays close to the raw sensor data size, and `02_preprocess.ipynb`
actively deletes any leftover `Out_pt*_dm/` folders from older runs.

### The regressed translation is replaced with a depth-measured one before ICP
The pose head's translation is regressed from a single globally-pooled feature vector — it never
sees *where* in the frame the object actually is, which made it the weakest part of the raw
prediction (~10 cm error). Before ICP refinement, `05_evaluate.ipynb` instead computes the
centroid of the real, depth-sensor-measured 3D points inside the object's mask and uses that as
the translation starting point (keeping the network's predicted rotation). This alone cut
Translation RMSE by roughly 4–5×, since it replaces a hard image-regression problem with a direct
geometric measurement.

### Two-stage ICP refinement
`refine_icp()` runs a coarse point-to-point pass (loose correspondence threshold, to capture large
initial offsets) followed by a finer point-to-plane pass (tight threshold, seeded by the coarse
result) against the observed depth point cloud — more accurate than a single point-to-point pass
alone, and the network's output is treated throughout as an *initial guess*, never scored directly.

### In-plane rotation augmentation was removed, not fixed
An earlier version rotated the RGB/depth/mask images for training augmentation but never rotated
the corresponding pose/radius-map targets to match, silently injecting up to ±10° of label noise
into every augmented sample. Computing the correct compensating rotation is a real fix but risks a
sign/convention error that's hard to catch without live testing; removing the rotation step (color
jitter + depth noise remain) is the safer trade — it costs a small amount of augmentation
diversity in exchange for zero risk of a *new*, harder-to-detect bug.

---

## 🚀 Quick Start (Google Colab)

Run the notebooks **in order**, in the same Colab session/runtime:

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_setup.ipynb` | Mount Drive, install packages, verify GPU, write `config.json`, clone this repo |
| 2 | `02_preprocess.ipynb` | Extract + preprocess each class end-to-end (poses, keypoints, splits) |
| 3 | `03_yolo_train.ipynb` | Train YOLOv8s for 2D object detection |
| 4 | `04_pose_train.ipynb` | Train EnhancedRCVPose — frozen warm-up → full fine-tune → rotation fine-tune |
| 5 | `05_evaluate.ipynb` | Validation + held-out test metrics (Translation RMSE, Rotation Error, ADD) |
| 6 | `06_visualize.ipynb` | Pose wireframe overlays, YOLO detections, radius-map heatmaps |

Every notebook reads `/content/config.json` (written by `01_setup.ipynb`) — no paths are
hardcoded or copy-pasted between notebooks.

**Local / non-Colab setup:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
The notebooks assume Colab-style paths (`/content/...`) and Google Drive mounting; adapt
`01_setup.ipynb`'s config cell if running elsewhere.

---

## 📉 Training strategy & loss

**Three stages** (`04_pose_train.ipynb`):

| Stage | Epochs | Backbone | LR schedule | Notes |
|---|---|---|---|---|
| 1 — Warm-up | 15 | frozen | `OneCycleLR`, peak `1e-3` | Only `pose_head`/`outside9_head` train; BatchNorm frozen too |
| 2 — Full fine-tune | up to 65 | unfrozen | fresh `OneCycleLR`, peak `3e-4` | Early stopping, `patience=15` |
| 3 — Rotation fine-tune | 10 | frozen again | `CosineAnnealingLR`, `1e-4` | Same rotation weight as main training |

`WeightedPoseLoss`:
```
L = W_TRANS · L_trans + W_ROT · L_rot + W_PTS · L_pts        (1.0, 10.0, 1.0)
```
- `L_trans` — MSE on `[tx, ty, tz]`
- `L_rot` — geodesic angle between the predicted (6-D-decoded) and ground-truth rotation matrices
- `L_pts` — MSE on the 9 predicted radius maps

Gradient accumulation, AMP (`autocast` + `GradScaler`), `pin_memory`, and `persistent_workers`
are used throughout; batch size / worker count auto-scale to the detected GPU (H100/A100/T4).

---

## 📐 Evaluation pipeline

`05_evaluate.ipynb`, per sample:
1. Run the network → decode the 9-D output to `[t, quaternion]`.
2. Replace `t` with the depth+mask centroid translation guess (see above).
3. Refine `[t, quaternion]` with two-stage ICP against the observed depth point cloud.
4. Score against ground truth: Translation RMSE, Rotation Error, Points MSE, ADD, ADD Success %.

ADD success uses **per-class thresholds** (`config.json`'s `ADD_THRESHOLDS`, standard LINEMOD
benchmark values) and mesh points loaded unscaled (matching this project's original reference
implementation), so results are directly comparable across runs.

---

## 🗃️ Dataset: LINEMOD

13 classes (2 of the standard 15 are excluded — texture-poor/ambiguous geometry):

`ape · benchvise · cam · can · cat · driller · duck · eggbox · glue · holepuncher · iron · lamp · phone`

**Split per class** (`02_preprocess.ipynb`, deterministic per-class seed): **70% train / 20% val
/ 10% test** — test is held out and should only be scored once at the end.

**Per-class layout after preprocessing:**
```
data/XX/
  rgb/          RGB images (.png)
  depth/        Depth images (.dpt — binary uint16, millimetres)
  pose/         Ground-truth [R|t] matrices (.npy, 3x4), translation in metres
  mask/         Object masks (.png)
  mesh.ply      3D model — ADD metric + visualization
  Outside9.npy  9 FPS-sampled 3D keypoints (mm) — radius maps derived from these on-the-fly
  Split/        train.txt / val.txt / test.txt
  gt.yml        Original ground-truth (kept for reference; not read after preprocessing)
```

---

## 📁 Project Structure

```
6D-pose-estimation/
├── 01_setup.ipynb          # Env setup, Drive mount, GPU check, config.json, repo clone
├── 02_preprocess.ipynb     # Per-class extraction, poses, keypoints, splits
├── 03_yolo_train.ipynb     # YOLOv8s detection training
├── 04_pose_train.ipynb     # EnhancedRCVPose training (3 stages)
├── 05_evaluate.ipynb       # Validation + test metrics
├── 06_visualize.ipynb      # Pose overlays, YOLO detections, radius-map heatmaps
├── src/
│   ├── model.py             # EnhancedRCVPose, WeightedPoseLoss, 6-D rotation helpers
│   └── dataset.py           # PoseDataset (on-the-fly radius maps), augmentation, safe_collate
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

All paths and shared settings live in `/content/config.json`, written by `01_setup.ipynb` and
extended by later notebooks (`YOLO_MODEL_PATH`, `BEST_POSE_MODEL`, …) as they produce artifacts:

```json
{
  "DATA_DIR": "/content/dataset/linemod/Linemod_preprocessed/data",
  "YOLO_DIR": "/content/dataset/linemod/Linemod_ready",
  "DRIVE_MODELS": "/content/drive/MyDrive/models",
  "REPO_DIR": "/content/6D-pose-estimation",
  "ALL_CLASSES": ["01","02","04","05","06","08","09","10","11","12","13","14","15"],
  "CLASS_NAMES": ["ape","benchvise","cam","can","cat","driller","duck","eggbox","glue","holepuncher","iron","lamp","phone"],
  "CAMERA_K": [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]],
  "ADD_THRESHOLDS": {"01": 0.01421, "02": 0.03309, "...": "..."},
  "YOLO_MODEL_PATH": "...Drive.../yolo_best.pt",
  "BEST_POSE_MODEL": "...Drive.../best_rcvpose_<timestamp>_finetuned.pth"
}
```

Re-running `01_setup.ipynb`'s config cell **merges** with any existing `config.json` rather than
overwriting it, so keys added later by other notebooks survive; `05_evaluate.ipynb` and
`06_visualize.ipynb` additionally auto-detect the newest checkpoint on Drive if `BEST_POSE_MODEL`
is ever missing, instead of failing to load a model.

---

## 📚 References

- **RCVPose** — Xu et al., *RCVPose: Recovery of 3D Pose from Radial Correspondences*
- **6-D rotation representation** — Zhou et al., *On the Continuity of Rotation Representations in Neural Networks* ([arXiv:1812.07035](https://arxiv.org/abs/1812.07035))
- **LINEMOD dataset** — Hinterstoisser et al., *Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes*
- **ADD metric** — Xiang et al., *PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes*
- **YOLOv8** — [Ultralytics](https://github.com/ultralytics/ultralytics)
