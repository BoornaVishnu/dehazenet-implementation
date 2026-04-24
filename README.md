# Dehazing for Outdoor Scenes in Automotive Applications

[![Repository](https://github.com/BoornaVishnu/dehazenet-implementation)

This project implements a deep learning–based framework for **single-image and video dehazing** focused on **outdoor automotive environments**, such as dashcam and on-road camera footage. The objective is to improve visibility in adverse weather conditions (fog, haze, smog) while preserving scene structure critical for downstream perception tasks.

The system is based on a **physics-inspired transmission estimation network (DehazeNet-style)** and supports inference on images, image batches, and MP4 videos.

---

# Dehazing for Outdoor Scenes in Automotive Applications

This project implements a deep learning–based framework for **single-image and video dehazing** focused on **outdoor automotive environments**, such as dashcam and on-road camera footage. The objective is to improve visibility in adverse weather conditions (fog, haze, smog) while preserving scene structure critical for downstream perception tasks.

The system is based on a **physics-inspired transmission estimation network (DehazeNet-style)** and supports inference on images, image batches, and MP4 videos.

---

## Project Overview

Atmospheric haze degrades image quality by reducing contrast and obscuring distant objects. This project estimates a **per-pixel transmission map** and reconstructs a clearer image using the atmospheric scattering model.

The pipeline supports:
- Training on hazy outdoor image datasets
- Inference on single images
- Batch inference on image folders
- Frame-by-frame video dehazing
- Side-by-side video visualization for qualitative analysis

The project is designed for **research and educational use**, with an emphasis on automotive vision applications.

---

## Repository Structure
.
├── data/                 # Input/Groundtruth pair dataset creation
├── models/               # DehazeNet model definition
├── utils/                # Utilities (I/O, reconstruction, visualization)
├── train.py              # Training script
├── eval.py               # Evaluation script
├── infer.py              # Single-image inference

## Installation

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (recommended for video I/O)
- Pillow (PIL)
- imageio (fallback for video processing)

Install dependencies:

```bash
pip install torch torchvision opencv-python pillow imageio
```

## Training

```
python train.py \
  --data_root "<path_to_training_dataset>" \
  --out_dir "<path_to_output_directory>" \
  --epochs 5 \
  --batch_size 16 \
  --amp
  --t0 0.1
```

## Single Image Inference

```
python infer.py \
  --ckpt "<path_to_checkpoint>/best.pt" \
  --input "<path_to_image>.jpg" \
  --img_size 256 \
  --t0 0.1
```
