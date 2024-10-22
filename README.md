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

## Technology Stack

- Python 3.x
- TensorFlow 2.x
- CUDA-enabled GPU with installed CUDA drivers
- OpenCV
- NumPy
- Matplotlib

## Dataset

The model is trained using the P3M-10k dataset, a large-scale portrait image dataset consisting of 10,000 images annotated with fine-level details for matting. This dataset ensures high-quality segmentation, making it suitable for real-world portrait matting applications.

For more details on the dataset, visit the official P3M-10k repository [here](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUh1MGRSM1hDYjN2Nzl4a2Vfc2xXNk5Dd0t1d3xBQ3Jtc0trQ3JTUUNJeEJ3Y2NVU3ZwdVFZa0VyNmZTTVJNbUJOVVJRdzRMRGhIdnJ4SXRGQWkxNGhKWUYzeUlaY2I2dWZVOHNCaTdDakdBc2V6aGluSzlKQzlBRXBQcFVCSUs0SG9NMzk2YUltTV9nTV9OQjhmbw&q=https%3A%2F%2Fdrive.google.com%2Fuc%3Fexport%3Ddownload%26id%3D1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1&v=S54EprtQdjA).

## References

- Paper: U^2-Net: [Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007).
- P3M-10k Dataset: Visit the dataset repository [here](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUh1MGRSM1hDYjN2Nzl4a2Vfc2xXNk5Dd0t1d3xBQ3Jtc0trQ3JTUUNJeEJ3Y2NVU3ZwdVFZa0VyNmZTTVJNbUJOVVJRdzRMRGhIdnJ4SXRGQWkxNGhKWUYzeUlaY2I2dWZVOHNCaTdDakdBc2V6aGluSzlKQzlBRXBQcFVCSUs0SG9NMzk2YUltTV9nTV9OQjhmbw&q=https%3A%2F%2Fdrive.google.com%2Fuc%3Fexport%3Ddownload%26id%3D1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1&v=S54EprtQdjA).
License
