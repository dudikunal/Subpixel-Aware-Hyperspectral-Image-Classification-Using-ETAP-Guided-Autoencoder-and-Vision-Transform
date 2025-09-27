# Knowledge-Embedded Representation Learning via Attribute Profiles and Transformer Fusion for Hyperspectral Image Classification

This repository provides the official PyTorch implementation of **Knowledge-Embedded Representation Learning via Attribute Profiles and Transformer Fusion**, a novel hybrid architecture for robust **Subpixel-Aware Hyperspectral Image (HSI) Classification**. This framework integrates spatial-spectral enhancement using **Extended Threshold-Free Attribute Profiles (eTAP)** with a dual-stream autoencoder and **Vision Transformer (ViT)** to capture both local unmixing cues and global contextual dependencies.

---

## 🧠 Key Features

Hyperspectral images (HSIs) often suffer from low spatial resolution, leading to the presence of **mixed pixels**—each pixel potentially containing multiple material signatures. Conventional methods often fail to exploit the full spatial-spectral correlation and subpixel information. This model addresses those challenges via:

- **ETAP-Guided Enhancement**: A pre-processing step enhances spatial features through Extended Threshold-Free Attribute Profiles, guiding the network with rich spatial priors.

- **Dual-Stream Autoencoder**: One stream focuses on **spectral unmixing**, estimating endmembers and abundances; the other focuses on **reconstruction-enhanced spectral learning**. Both streams jointly model subpixel mixing effects.

- **Vision Transformer Fusion**: A ViT module captures long-range spectral-spatial dependencies, enabling robust global reasoning over the hyperspectral volume.

- **Fusion Strategy**: The dual-stream features and ViT embeddings are synergistically fused for final classification, combining subpixel precision with contextual awareness.

---

## 📂 Repository Structure
```
.
├── data/
│   ├── indian_pines_TAP.mat
│   ├── salinasTAP15PC.mat
│   ├── Pavia_30.mat
│   └── ... (other .mat files as needed)
├── results/
│   └── (model checkpoints and logs saved here)
├── dataset.py
├── utils.py
├── model.py
├── demo.py
└── README.md
```


- **dataset.py** → Handles data loading, preparation, normalization, mirror-padding, and dataset splitting for training/testing.  
- **utils.py** → Contains utility functions for performance metric calculations, logging, and helper classes.  
- **model.py** → Defines the architecture:  
  - Sub-pixel unmixing module (autoencoder–style network)  
  - Vision Transformer classifier  
  - Fusion of unmixing outputs and transformer embeddings  
- **demo.py** → Script for training and testing the model. Includes main training loop, validation, saving, and result reporting.  

---

## ⚙️ Dependencies

Ensure the following are installed:

- Python 3.7+  
- PyTorch (>= 1.7)  
- NumPy  
- SciPy  
- scikit-learn  
- Transformers (for the ViT module)  
- Matplotlib (optional, for visualizations)  

---

