# Team - RnD

# Automatic Image Colorization using Ensemble of Deep Convolutional Neural Networks!

This repository provides an advanced solution for the automatic colorization of grayscale images using deep learning techniques. The project aims to transform black-and-white images into vibrant, colorized versions, leveraging cutting-edge neural network architectures and pre-trained models. The main goal is to enhance the visual appeal and utility of grayscale imagery in applications such as media restoration, photography enhancement, and historical archives.

## Table of Contents

1. [Introduction](#introduction) 

2. [Previous Work](#previous-work)

3. [Data Collection](#data-collection)  
   3.1 [CIFAR-10 Dataset](#cifar-10-dataset)  
   3.2 [DIV2K Dataset](#div2k-dataset)

4. [Data Preprocessing](#data-preprocessing)  
   4.1 [Resizing to 224x224](#resizing-to-224x224)  
   4.2 [Enhancing Low-Resolution Images](#enhancing-low-resolution-images)  
   4.3 [LAB Color Space Conversion](#lab-color-space-conversion)  
   4.4 [Channel Normalization](#channel-normalization)  
   4.5 [Dataset Splitting](#dataset-splitting)

5. [Model Architecture](#model-architecture)  
   5.1 [Automatic Colorization Model](#automatic-colorization-model)  
   5.2 [Real-ESRGAN+ Pipeline](#real-esrgan-pipeline)

6. [Model Training](#model-training)

7. [Model Evaluation](#model-evaluation)

8. [Results](#results)

9. [Contributors](#contributors)

10. [Credits](#credits)


---

## Introduction

This project focuses on the automatic colorization of grayscale images, combining advanced deep learning techniques and innovative preprocessing strategies. The model architecture incorporates ResNet50 and DenseNet121 for feature extraction, fusion blocks for effective feature merging, and decoder blocks for high-quality reconstruction. To ensure input images meet the required resolution of 224x224 pixels, we have utilized Real-ESRGAN+ for super-resolution enhancement. This step generates a new, high-quality dataset from smaller images, enabling the model to maintain superior colorization performance. The project aims to restore vibrancy and realism to grayscale images, with potential applications in media restoration, photography, and beyond.

---

## Previous Work

Previous studies on automatic colorization have explored various architectures, with many focusing on models that utilize feature extractors like ResNet and DenseNet alongside decoder networks. However, a significant limitation of these studies was the unavailability of open-source implementations for such architectures, making replication and further innovation challenging.

To address this gap, this project develops the same architecture that combines ResNet50 and DenseNet121 for feature extraction, fusion blocks for feature merging, and decoder blocks for high-quality colorization. The architecture is carefully designed to balance computational efficiency and output quality, enabling it to achieve competitive results in image colorization tasks.

---

## Data Collection

This project utilizes two datasets: CIFAR-10 and DIV2K, each offering unique characteristics that enhance the training and evaluation process for automatic colorization.

### 3.1 CIFAR-10 Dataset

A widely used dataset in computer vision, CIFAR-10 consists of 60,000 low-resolution (32x32 pixels) images across 10 distinct classes. It serves as a diverse dataset, allowing the model to learn general colorization patterns across various categories like animals, vehicles, and objects.

### 3.2 DIV2K Dataset

This high-resolution dataset is designed for super-resolution tasks. It contains 1,000 high-quality images with resolutions up to 2K, making it ideal for training the preprocessing pipeline and fine-tuning the model on high-detail scenarios.

These datasets together provide a robust foundation for training and testing the model, enabling it to handle a variety of image resolutions and visual complexities. Furthermore, images smaller than 224x224 (CIFAR-10) are enhanced using Real-ESRGAN+ to ensure consistent quality and resolution before colorization.

---

## Data Preprocessing

Data preprocessing is a crucial step in ensuring the quality and consistency of input data for the automatic colorization model. The following steps were implemented:

### 4.1 Resizing to 224x224

All images were resized to a uniform resolution of 224x224 pixels, ensuring compatibility with the model's input requirements.

### 4.2 Enhancing Low-Resolution Images

For images with resolutions smaller than 224x224, a new dataset was created using Real-ESRGAN+ to enhance clarity and detail before resizing. This ensures that even low-quality images contribute effectively to the training process.

### 4.3 LAB Color Space Conversion

All RGB images were converted to the CIE LAB color space. This transformation allows the model to predict only the a and b color channels while taking the L (lightness) channel as input from the grayscale image.

### 4.4 Channel Normalization

- The **L channel** was normalized to a range of [0, 1] for stable input to the model.
- The **a and b channels** were scaled to a standard range to facilitate efficient and balanced learning during training.

### 4.5 Dataset Splitting

The dataset was split into training, and test sets to ensure unbiased evaluation and effective tuning of model parameters.

---

## Model Architecture

The architecture for this project is specifically designed to address the challenges of automatic colorization by leveraging deep learning techniques and advanced feature extraction mechanisms. Below is an overview of the main components of the model architecture, followed by the preprocessing pipeline utilizing Real-ESRGAN+.

### 5.1 Automatic Colorization Model

The model consists of the following key elements:

- **Feature Extraction**:  
  ResNet50 and DenseNet121 are used as dual feature extractors. These architectures extract rich and diverse features from the grayscale L channel input, enabling the model to effectively capture spatial and contextual information.

- **Fusion Blocks**:  
  Four fusion blocks combine features from ResNet50 and DenseNet121, merging their outputs to create a comprehensive feature representation. Skip connections are incorporated to preserve low-level details and enhance gradient flow.

- **Decoder Blocks**:  
  The decoder network reconstructs the a and b channels from the fused features. Each decoder block consists of 2D convolutions, batch normalization, ReLU activation, and upsampling layers to restore the resolution. Skip connections link specific fusion blocks to decoder blocks for better reconstruction of finer details.

- **Output**:  
  The model predicts the a and b channels in the LAB color space. These outputs, combined with the original L channel input, produce the final colorized image.

A visual representation of the model architecture is included below.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/e9bd0990-abf4-4eeb-9152-b4d929c6be55" width="800" />
</p>

### 5.2 Real-ESRGAN+ Pipeline

To enhance the clarity and quality of low-resolution images before feeding them into the colorization model, Real-ESRGAN+ is integrated into the preprocessing pipeline. This step ensures that:

- Images smaller than 224x224 are enhanced and upscaled while preserving details.
- The enhanced images are subsequently resized to 224x224, maintaining consistency across the dataset.

The Real-ESRGAN+ pipeline complements the automatic colorization model by ensuring high-quality inputs, thereby improving the overall performance of the system. This model is implicitly integrated to the existing model. All credits goes to open source python libraries related to Real-ESRGAN+ -> basicsr, realesrgan.

A visual representation of the Real-ESRGAN+ architecture is included below.  
<p align="center">
  <img src="https://github.com/user-attachments/assets/e61a389d-1ede-40bf-87dd-06257d9265a5" width="800" />
</p>

---


## Model Training

Model training is an essential phase where the model learns to colorize grayscale images. In this project, the model is trained using the preprocessed dataset, consisting of grayscale images (L channel) as input and corresponding colorized outputs (a and b channels) in the LAB color space. The following steps and hyperparameters were used during training:

### 6.1 Hyperparameters

The model was trained using the following hyperparameters, which were selected based on previous research and optimized for this task:

- **Batch Size**: 8  
  A batch size of 8 was chosen to strike a balance between training speed and stability.
  
- **Learning Rate**: 0.01  
  The learning rate was set at 0.01 to allow the model to converge efficiently without overshooting the optimal parameters.

- **Epochs**: 3  
  The model was trained for 3 epochs. This limited number of epochs was sufficient to demonstrate the effectiveness of the architecture, with running and validation loss being tracked for performance evaluation.

- **Loss Function**:  
  The model used L2 loss (mean squared error) for pixel-wise differences.

- **Optimizer**: Adam Optimizer  
  The Adam optimizer was used for weight updates, as it adapts the learning rate during training and improves convergence speed.

### 6.2 Training Process

1. **Input Data**:  
   The preprocessed dataset (after resizing, enhancement, LAB color space conversion, and channel normalization) was fed into the model in batches of 8 images.
   
2. **Model Training**:  
   The model was trained using the Adam optimizer with a learning rate of 0.01 for 3 epochs. During each epoch, the model learned to predict the a and b color channels from the grayscale L channel input.

3. **Loss Tracking**:  
   The running loss (training loss) and validation loss were tracked throughout the training process. This helped monitor the model’s performance and detect overfitting.


---


## Model Evaluation

The performance of the automatic colorization model was evaluated using several key metrics that assess both the accuracy of the colorization and the perceptual quality of the generated images. The following metrics were used to evaluate the model:

- **Mean Squared Error (MSE)**:  
  This metric measures the average squared differences between the predicted color values and the ground truth. A lower MSE indicates better colorization performance.

- **Peak Signal-to-Noise Ratio (PSNR)**:  
  PSNR is used to evaluate the quality of the colorized images. It compares the original and predicted images to measure how much noise is present, with higher PSNR values indicating better image quality.

- **Structural Similarity Index (SSIM)**:  
  SSIM is a perceptual metric that evaluates the similarity between the predicted and ground truth images in terms of luminance, contrast, and structure. Higher SSIM values suggest that the colorized images maintain more structural similarity with the original color images.

- **Learned Perceptual Image Patch Similarity (LPIPS)**:  
  LPIPS measures the perceptual similarity between the predicted and ground truth images. Unlike pixel-wise metrics like MSE, LPIPS evaluates the similarity based on high-level features, offering a better reflection of human perception. The metric was calculated using the APEX model for high-level feature extraction. Lower LPIPS scores indicate better perceptual quality of the colorized images.

These metrics provide a comprehensive evaluation of both the quantitative and qualitative aspects of the colorization model, ensuring that the generated results are both accurate and visually appealing.

---


## Results

The performance of the automatic colorization model was evaluated on a variety of test images. Below are example of respective colorized image along with its corresponding evaluation metrics.

- **CIFAR10 (Concatenation Fusion Technique)**:
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/434c5599-76cf-4a80-b347-c87d54119815" width="700" />
</p>

  | Metric             | Value     |
  |--------------------|-----------|
  | **MSE**            | 0.003     |
  | **PSNR**           | 24.8253 dB  |
  | **SSIM**           | 0.9554      |
  | **LPIPS**          | 0.1236     |

- **CIFAR10 (Average Fusion Technique)**:
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/336da002-ebe7-44dd-a652-d08fee4376b1" width="700" />
</p>

  | Metric             | Value     |
  |--------------------|-----------|
  | **MSE**            | 0.004    |
  | **PSNR**           | 24.8119 dB  |
  | **SSIM**           | 0.9537      |
  | **LPIPS**          | 0.1227     |

- **CIFAR10 (Max Fusion Technique)**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b4d2428-647f-4e2f-bd21-49fe2d0498a1" width="700" />
</p>

  | Metric             | Value     |
  |--------------------|-----------|
  | **MSE**            | 0.004     |
  | **PSNR**           | 24.7831 dB  |
  | **SSIM**           | 0.9544     |
  | **LPIPS**          | 0.1257     |

### The Concatenation fusion technique yields slightly better results, making it our preferred fusion strategy for the remaining work.

- **DIV2K (Concatenation)**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/26bfc0ce-6362-4834-9a02-667c5016d123" width="700" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/51ba9d70-3389-4d3d-bf83-0ae2561c728c" width="700" />
</p>

  | Metric             | Value     |
  |--------------------|-----------|
  | **MSE**            | 0.024     |
  | **PSNR**           | 27.9530 dB  |
  | **SSIM**           | 0.9119      |
  | **LPIPS**          | 0.0735     |

- **CIFAR10 (Real-ESRGAN+ Enhanced Dataset - Concatenation)**:

<p align="center">
  <img src="https://github.com/user-attachments/assets/405f4976-e8c9-43e3-bdae-82d58bf1ab98" width="700" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ada1e981-d8f0-44c1-8733-8449ed7e4aae" width="700" />
</p>

  | Metric             | Value     |
  |--------------------|-----------|
  | **MSE**            | 0.0017    |
  | **PSNR**           | 28.2752 dB  |
  | **SSIM**           | 0.9006      |
  | **LPIPS**          | 0.0840    |


We initially experimented with different fusion strategies—concatenation, average, and max—to determine which provided the best results. Our findings revealed that the concatenation technique outperformed the others when applied to the 32x32 low-resolution CIFAR-10 dataset, leading us to choose it as our fusion strategy for the remaining work. The DIV2K dataset, on the other hand, delivered promising results due to its high resolution. Additionally, enhancing the CIFAR-10 dataset using Real-ESRGAN+ significantly improved the colorization performance, showing superior results compared to the original 32x32 low-resolution CIFAR-10. Based on these observations, we concluded that for low-resolution images, enhancing the dataset first yields better results before applying the fusion strategy.

---

## Contributors

- *Nitesh Singh Bhadouria:*
   - Model – Max and Average (CIFAR10)
   - Concatenation (DIV2K)
   - Model working and Training (DIV2K)
   - Testing (PIMS,LPIPS, SSIMS).
  
- *Dhammadip Kamble:*
   - Model – Concatenation (CIFAR10)
   - Model working and Training (CIFAR10 and DIV2K)
   - Image Enhancement (Real-ESRGAN+)

- *Rahul Kumar:*
   - Data Preprocessing
   - Model – Max and Average (CIFAR10)
   - Testing (DIV2K : PIMS,LPIPS, SSIMS)


---



## Credits

We would like to acknowledge the following papers and authors whose work contributed to the development of this project:

1. **Paper Name**: *[Automatic Image Colorization using Ensemble of Deep Convolutional Neural Networks]*  
   **Authors**: [Urvi Oza, Arpit Pipara, Srimanta Mandal, Pankaj Kumar]  
   **Link to Paper**: [Link to Paper](https://ieeexplore.ieee.org/document/9864479)

2. **Paper Name**: *[Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data]*  
   **Authors**: [Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan]  
   **Link to Paper**: [Link to Paper](https://arxiv.org/pdf/2107.10833)
