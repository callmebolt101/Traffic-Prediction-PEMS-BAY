# Traffic Prediction using GMAN and GraphWaveNet

This repository provides a comparison of **GMAN** and **GraphWaveNet** models for traffic prediction using the **PEMS-BAY** dataset. The objective is to evaluate the performance of both models in forecasting traffic flow across multiple sensors over time.

---

## Table of Contents

1. [Overview](#overview)
2. [Models Used](#models-used)
3. [Dataset](#dataset)
4. [Dependencies](#dependencies)
5. [Instructions](#instructions)
   - [GMAN](#gman)
   - [GraphWaveNet](#graphwavenet)

---

## Overview

Traffic prediction plays a critical role in intelligent transportation systems. This project implements two advanced graph-based models:

1. **GMAN**: Graph Multi-Attention Network for spatio-temporal prediction.  
2. **GraphWaveNet**: A deep learning model that leverages graph convolution and wavelet transformation for traffic forecasting.

Both models are evaluated on the **PEMS-BAY** dataset.

---

## Models Used

1. **GMAN**:
   - A spatio-temporal attention model that captures spatial and temporal correlations using multi-head attention mechanisms.
   - Requires spatial embeddings to be generated before training.

2. **GraphWaveNet**:
   - Utilizes graph convolution to model spatial dependencies and temporal convolution to model time-series data.

---

## Dataset

We use the **PEMS-BAY** dataset for traffic flow prediction:
- **Source**: [PEMS-BAY Dataset](https://zenodo.org/records/4263971#.Yt5GCOxKj0o)
- Contains traffic sensor data with 325 nodes and multiple timestamps.

---

## Dependencies

Install the required Python libraries using the following command:

```bash
pip install torch torchvision numpy pandas scikit-learn networkx gensim
```

Ensure your environment has access to a GPU for faster training, such as on **Google Colab**.

---

## Instructions

### GMAN

Follow these steps to run the GMAN model:

1. **Generate Spatial Embeddings**  
   Run the following script to generate spatial embeddings using `node2vec`:

   ```bash
   python GMAN_generateSE.py
   ```

   This generates the spatial embeddings file required for the GMAN model.

2. **Train and Test GMAN**  
   After generating spatial embeddings, run the following script to train and evaluate the model:

   ```bash
   python pred_GMAN.py
   ```

   - Training logs, model checkpoints, and predictions will be saved in the `save/` directory.
   - The script uses Mean Absolute Error (MAE) as the default loss.

---

### GraphWaveNet

Follow these steps to run the GraphWaveNet model:



---
