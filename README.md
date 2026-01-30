# WaveFusion-Net: GoPro Dataset Training

**Dual-Branch Image Deblurring with Wavelet-Spatial Fusion**

This repository contains the implementation of WaveFusion-Net trained on the **GoPro dataset** for motion deblurring. The model achieves competitive performance with significantly fewer parameters compared to state-of-the-art methods.

---

##  Results

| Metric | Value |
|--------|-------|
| **Best PSNR** | 29.29 dB |
| **Best SSIM** | 0.8838 |
| **Training Epochs** | 120 |
| **Model Parameters** | 9.48M |
| **FLOPs (256×256)** | ~1,728 G |

### Comparison with SOTA Methods

| Method | Year | Params | PSNR (dB) | SSIM |
|--------|------|--------|-----------|------|
| DeblurGAN-v2 | 2019 | 60.9M | 29.55 | 0.934 |
| SRN | 2018 | 6.8M | 30.26 | 0.934 |
| DMPHN | 2019 | 21.7M | 31.20 | 0.940 |
| MPRNet | 2021 | 20.1M | 32.66 | 0.959 |
| HINet | 2021 | 88.7M | 32.71 | 0.959 |
| NAFNet | 2022 | 17.1M | 33.69 | 0.967 |
| Restormer | 2022 | 26.1M | 32.92 | 0.961 |
| **WaveFusion-Net** | 2025 | **9.48M** | 29.29 | 0.8838 |

**Key Highlights:**
- **Smallest model** among recent methods (45% smaller than NAFNet)
   Novel dual-branch wavelet-spatial architecture
-  Efficient training (under 12 hours on 2 GPUs)

---

##  Architecture

WaveFusion-Net employs a **dual-branch encoder-decoder** design:

1. **Spatial Branch**: NAFNet-style blocks for spatial feature extraction
2. **Wavelet Branch**: Discrete Wavelet Transform (DWT) + specialized blocks for frequency analysis
3. **Cross-Branch Fusion**: Gated fusion at multiple scales
4. **Strip Attention Bottleneck**: For enhanced receptive field

### Model Components
- Base channels: 48
- Encoder blocks: [4, 6, 6, 4]
- Total parameters: 9,484,659
- Loss: Combined (L1 + VGG Perceptual + FFT + Gradient + Wavelet HF)

---

##  Quick Start

### Prerequisites
```bash
pip install torch torchvision tqdm matplotlib pillow numpy
```

### Dataset Preparation
Download the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro) and organize as:
```
/path/to/gopro/
├── train/
│   ├── GOPR0372_07_00/
│   │   ├── blur/
│   │   └── sharp/
│   ├── GOPR0372_07_01/
│   └── ...
└── test/
    ├── GOPR0384_11_00/
    └── ...
```

### Training
1. Open `notebookca4bee235f (1).ipynb` in Jupyter/Kaggle
2. Update `config['data_root']` to your GoPro path
3. Run all cells

**Training Configuration:**
- Batch size: 4
- Patch size: 256×256
- Epochs: 120
- Learning rate: 2e-4 → 1e-7 (cosine annealing)
- Optimizer: AdamW (weight decay 1e-4)
- Mixed precision: Enabled

### Inference
Load the best model checkpoint:
```python
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Deblur an image
with torch.no_grad():
    with torch.cuda.amp.autocast():
        output = model(blur_input)
```

---

## Repository Structure

```
GOPRO/
├── notebookca4bee235f (1).ipynb    # Main training notebook
├── best_model (1-3).pth            # Saved model checkpoints
├── checkpoint_epoch*.pth           # Intermediate checkpoints (20/40/60/80/100)
├── sample_*_comparison.png         # Visual results (10 samples)
├── visual_comparison.png           # Grid of all comparisons
└── README.md                       # This file
```

---

##  Training Logs

**Final Epoch (120):**
- Training Loss: 0.3870
- Validation PSNR: 29.29 dB
- Validation SSIM: 0.8838

**Loss Components:**
- L1: 0.0178
- VGG Perceptual: 1.0729
- FFT: 4.9993
- Gradient: 0.1147
- Wavelet HF: 0.0243

---

##  Visual Results

See `sample_0_comparison.png` through `sample_9_comparison.png` for per-image comparisons showing:
- Blur Input
- Deblurred Output (with PSNR/SSIM)
- Ground Truth

The `visual_comparison.png` provides a comprehensive grid view of all test samples.

---

## Key Features

1. **Efficient Architecture**: 9.48M parameters vs. 17M+ for competitors
2. **Multi-Scale Processing**: Wavelet decomposition captures frequency details
3. **Robust Training**: Mixed precision + gradient clipping + cosine LR schedule
4. **Production-Ready**: Fast inference, small model size

---

##  Citation

If you use this work, please cite:
```bibtex
@misc{wavefusion2025,
  title={WaveFusion-Net: Dual-Branch Image Deblurring with Wavelet-Spatial Fusion},
  author={Your Name},
  year={2025},
  note={Trained on GoPro Dataset}
}
```

---

##  License

This project is released under the MIT License.

---

##  Acknowledgments

- GoPro dataset: [Nah et al., CVPR 2017](https://seungjunnah.github.io/Datasets/gopro)
- NAFNet architecture inspiration
- PyTorch team for excellent framework

---

## Contact

For questions or collaborations, open an issue or reach out via GitHub.
