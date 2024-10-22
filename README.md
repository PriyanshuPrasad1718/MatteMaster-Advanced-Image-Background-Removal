# U2-Net Image Matting for High-Quality Background Removal

This project implements the **U2-Net architecture** in TensorFlow, enabling high-quality background removal from human portraits. It leverages the advancements of deep learning to perform **image matting** with remarkable precision, producing polished and professional results.

## Overview

U2-Net, proposed in the paper [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007), offers an advanced approach to background removal by employing a nested U-structure that captures both fine and coarse details of the subject. This architecture allows for **high-quality segmentation**, which makes it ideal for background removal tasks, especially with complex human portraits.

For training, we utilized the **P3M-10k dataset** – a large-scale dataset designed specifically for human portrait matting tasks, containing **10,000 finely annotated images**. The dataset's high quality contributes to the accurate and reliable background removal achieved in this project.

The training was run for **10 epochs**, taking approximately **18 hours** with the help of **CUDA** to accelerate processing on the GPU.

## Features

- **High-Quality Background Removal**: Achieve professional-grade background removal from human portraits.
- **U2-Net Architecture**: Implementation based on the U2-Net paper to capture fine details with precision.
- **P3M-10k Dataset**: Large-scale dataset with 10,000 finely annotated human portraits for accurate image matting.
- **CUDA Support**: Utilizes GPU processing for faster training and inference.
- **TensorFlow Support**: Built using TensorFlow for flexibility and efficiency.
- **Python Compatibility**: Written in Python, making it accessible and easy to integrate with other projects.

## Getting Started

### Technology Stack

- Python 3.x
- TensorFlow 2.x
- CUDA-enabled GPU with installed CUDA drivers
- OpenCV
- NumPy
- Matplotlib

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/U2Net-Image-Matting.git
cd U2Net-Image-Matting
