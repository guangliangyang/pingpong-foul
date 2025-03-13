# Foul Detection for Table Tennis Serves Using Deep Learning

This repository contains the implementation code and resources for the research paper:

**"Foul Detection for Table Tennis Serves Using Deep Learning"**  
*Authors: Guang Liang Yang, Minh Nguyen, Wei Qi Yan, Xue Jun Li*  
*Published in Electronics, 2025*  
[https://www.mdpi.com/2079-9292/14/1/27](https://www.mdpi.com/2079-9292/14/1/27)

## Overview

This project focuses on developing an automated system to detect serve fouls in table tennis using deep learning techniques. By analyzing 3D ball trajectories captured through a multi-camera setup, the system employs YOLO models for ball detection and Transformer models for identifying critical trajectory points. The goal is to ensure fair play by accurately detecting fouls such as improper ball toss height and incorrect positioning.

## Repository Structure

- **`data/`**: Contains the dataset used for training and evaluation.
- **`models/`**: Pre-trained models and scripts for training YOLO and Transformer models.
- **`scripts/`**: Utility scripts for data preprocessing, model training, and evaluation.
- **`results/`**: Outputs from model evaluations, including metrics and visualizations.
- **`docs/`**: Additional documentation and resources.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- OpenCV
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/guangliangyang/table-tennis-foul-detection.git
   cd table-tennis-foul-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Data Preparation**: Instructions on how to prepare and format your dataset.
- **Model Training**: Steps to train the YOLO and Transformer models.
- **Evaluation**: Guidelines to evaluate the performance of the models on your dataset.

## Results

The system achieved the following performance metrics:

- **YOLO-based Ball Detection**:
  - Precision: 87.52%
  - Recall: 83.37%

- **Transformer-based Key Point Identification**:
  - F1 Score: 93%

These results demonstrate the system's capability to accurately detect serve fouls in table tennis matches.

## Citation

If you find this work useful, please cite our paper:

Yang, G.L.; Nguyen, M.; Yan, W.Q.; Li, X.J. Foul Detection for Table Tennis Serves Using Deep Learning. *Electronics* **2025**, *14*, 27. [https://doi.org/10.3390/electronics14010027](https://doi.org/10.3390/electronics14010027)

