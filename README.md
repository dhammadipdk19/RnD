# Team - RnD

# Automatic Image Colorization using Ensemble of Deep Convolutional Neural Networks!

This repository provides an advanced solution for the automatic colorization of grayscale images using deep learning techniques. The project aims to transform black-and-white images into vibrant, colorized versions, leveraging cutting-edge neural network architectures and pre-trained models. The main goal is to enhance the visual appeal and utility of grayscale imagery in applications such as media restoration, photography enhancement, and historical archives.

## Table of Contents
1. [Introduction](#introduction)
2. [Previous Work](#previous-work)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Comparison Across Models](#comparison-across-models)
9. [Error Analysis](#error-analysis)
10. [Contributors](#contributors)


## Introduction

This project focuses on the automatic colorization of grayscale images, combining advanced deep learning techniques and innovative preprocessing strategies. The model architecture incorporates ResNet50 and DenseNet121 for feature extraction, fusion blocks for effective feature merging, and decoder blocks for high-quality reconstruction. To ensure input images meet the required resolution of 224x224 pixels, we have utilized Real-ESRGAN+ for super-resolution enhancement. This step generates a new, high-quality dataset from smaller images, enabling the model to maintain superior colorization performance. The project aims to restore vibrancy and realism to grayscale images, with potential applications in media restoration, photography, and beyond.

## Previous Work

Previous studies on automatic colorization have explored various architectures, with many focusing on models that utilize feature extractors like ResNet and DenseNet alongside decoder networks. However, a significant limitation of these studies was the unavailability of open-source implementations for such architectures, making replication and further innovation challenging.

To address this gap, this project develops the same architecture that combines ResNet50 and DenseNet121 for feature extraction, fusion blocks for feature merging, and decoder blocks for high-quality colorization. The architecture is carefully designed to balance computational efficiency and output quality, enabling it to achieve competitive results in image colorization tasks.

## Data Collection

This project utilizes two datasets: CIFAR-10 and DIV2K, each offering unique characteristics that enhance the training and evaluation process for automatic colorization.

#### CIFAR-10:
A widely used dataset in computer vision, CIFAR-10 consists of 60,000 low-resolution (32x32 pixels) images across 10 distinct classes. It serves as a diverse dataset, allowing the model to learn general colorization patterns across various categories like animals, vehicles, and objects.

#### DIV2K:
This high-resolution dataset is designed for super-resolution tasks. It contains 1,000 high-quality images with resolutions up to 2K, making it ideal for training the preprocessing pipeline and fine-tuning the model on high-detail scenarios.

These datasets together provide a robust foundation for training and testing the model, enabling it to handle a variety of image resolutions and visual complexities. Furthermore, images smaller than 224x224 (CIFAR10) are enhanced using Real-ESRGAN+ to ensure consistent quality and resolution before colorization.

## Data Preprocessing

Data preprocessing is a crucial step in ensuring the quality and consistency of input data for the automatic colorization model. The following steps were implemented:

*Resizing to 224x224*:
All images were resized to a uniform resolution of 224x224 pixels, ensuring compatibility with the model's input requirements.

*Enhancing Low-Resolution Images*:
For images with resolutions smaller than 224x224, a new dataset was created using Real-ESRGAN+ to enhance clarity and detail before resizing. This ensures that even low-quality images contribute effectively to the training process.

*Conversion to LAB Color Space*:
All RGB images were converted to the CIE LAB color space. This transformation allows the model to predict only the a and b color channels while taking the L (lightness) channel as input from the grayscale image.

*Channel Normalization*:
The L channel was normalized to a range of [0, 1] for stable input to the model.
The a and b channels were scaled to a standard range to facilitate efficient and balanced learning during training.

*Splitting into Train, Validation, and Test Sets*:
The dataset was split into training, validation, and test sets to ensure unbiased evaluation and effective tuning of model parameters.

## Model Architecture

The architecture for this project is specifically designed to address the challenges of automatic colorization by leveraging deep learning techniques and advanced feature extraction mechanisms. Below is an overview of the main components of the model architecture, followed by the preprocessing pipeline utilizing Real-ESRGAN+.

### Automatic Colorization Model
The model consists of the following key elements:

*Feature Extraction*:
ResNet50 and DenseNet121 are used as dual feature extractors. These architectures extract rich and diverse features from the grayscale L channel input, enabling the model to effectively capture spatial and contextual information.

*Fusion Blocks*:
Four fusion blocks combine features from ResNet50 and DenseNet121, merging their outputs to create a comprehensive feature representation.
Skip connections are incorporated to preserve low-level details and enhance gradient flow.

*Decoder Blocks*:
The decoder network reconstructs the a and b channels from the fused features.
Each decoder block consists of 2D convolutions, batch normalization, ReLU activation, and upsampling layers to restore the resolution.
Skip connections link specific fusion blocks to decoder blocks for better reconstruction of finer details.

*Output*:
The model predicts the a and b channels in the LAB color space. These outputs, combined with the original L channel input, produce the final colorized image.

A visual representation of the model architecture is included below.
![image](https://github.com/user-attachments/assets/e9bd0990-abf4-4eeb-9152-b4d929c6be55)

### Real-ESRGAN+ Pipeline
To enhance the clarity and quality of low-resolution images before feeding them into the colorization model, Real-ESRGAN+ is integrated into the preprocessing pipeline. This step ensures that:

Images smaller than 224x224 are enhanced and upscaled while preserving details.
The enhanced images are subsequently resized to 224x224, maintaining consistency across the dataset.
The Real-ESRGAN+ pipeline complements the automatic colorization model by ensuring high-quality inputs, thereby improving the overall performance of the system.

A visual representation of the Real-ESRGAN+ architecture is included below.
![image](https://github.com/user-attachments/assets/e61a389d-1ede-40bf-87dd-06257d9265a5)



## Model Training

Each model was trained using the preprocessed data. The hyperparameters for each model were tuned through techniques like grid search or random search to optimize performance. This includes adjusting the number of layers, the number of units in each layer, the learning rate, batch size, and other relevant parameters.

## Model Evaluation

The performance of each model was evaluated using the following metrics:

- *R² Score:* Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
- *Root Mean Squared Error (RMSE):* A metric that indicates the average magnitude of error in the model's predictions.
- *Mean Absolute Error (MAE):* Measures the average absolute difference between predicted and actual values.

## Comparison Across Models


| Model                        | MAE      | MSE              | R²     |
|------------------------------|----------|------------------|--------|
| *LSTM*                     | 2110.10  | 7.42e+06         | 0.9826 |
| *RNN*                       | 2503.22  | 1.02e+07         | 0.9761 |
| *GRU*                       | 2968.32  | 1.40e+07         | 0.9671 |
| *Bidirectional LSTM*        | 1792.30  | 5.76e+06         | 0.9865 |
| *LSTM with Attention*       | 2110.10  | 7.42e+06         | 0.9826 |
| *Stacked LSTM*              | 2042.55  | 6.92e+06         | 0.9838 |
| *Basic Transformer*         | 1393.69  | 3.91e+06         | 0.9908 |
| *Transformer with LSTM*     | 1310.49  | 3.22e+06         | 0.9924 |

The results show that the *Transformer with LSTM* model outperforms all other models, achieving the lowest MAE (1310.49), MSE (3.22e+06), and the highest R² score (0.9924).

## Error Analysis

Error analysis was performed to understand the types of errors each model makes. For instance:
- Some models may overestimate or underestimate electricity demand during peak hours or on weekends.
- The models may exhibit bias toward certain time periods due to the temporal nature of the data.

Understanding these errors allows for model improvements, such as adjusting the features or modifying model parameters.

## Contributors

- *Vissapragada Sandeep:* Data visualisation, Models Contributed - GRU, Transformer, and Bidirectional LSTM models.
- *Rohan Naskar:* Focused on Transformer model architecture and data preprocessing.
- *Saurabh Ramdas Nevase:* Worked on LSTM model and data preprocessing techniques.
- *Shrashank Maravi:* Contributed to the XGBoost model and Stacked LSTM model.
- *Sugandh Kumar:* Focused on RNN and LSTM models.


---

Feel free to clone this repository, run the models, and explore the code to improve and customize your own short-term electricity demand forecasting solutions!


## Work based on

1. [Urvi Oza], [Arpit Pipara], [Srimanta Mandal], and [Pankaj Kumar],
   (https://ieeexplore.ieee.org/document/9864479)
   **Automatic Image Colorization using Ensemble of Deep Convolutional Neural Networks**
