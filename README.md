# Human Activity Recognition with Hidden Markov Models

A machine learning project that recognizes human activities (walking, jumping, standing, and being still) using smartphone accelerometer and gyroscope data. The project employs **Hidden Markov Models (HMMs)** with the Baum–Welch algorithm for training and the Viterbi algorithm for sequence decoding.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Notebooks Guide](#notebooks-guide)
- [Requirements](#requirements)
- [Future Improvements](#future-improvements)

## Overview

This project demonstrates how probabilistic models can capture temporal dynamics in motion sensor data for reliable activity recognition. The pipeline processes raw accelerometer and gyroscope signals from a smartphone, extracts meaningful features, and trains an HMM to classify four distinct activities with high accuracy.

**Key Achievement**: Achieved **98.7% accuracy** on held-out test data using a probabilistic sequence model.

## Project Structure

```
Hidden-Markov-Model/
├── notebooks/
│   ├── data_collection_and_preprocessing.ipynb  # Data loading & cleaning
│   ├── feature-extraction.ipynb                  # Feature engineering
│   └── hmm-model.ipynb                           # HMM training & evaluation
├── processed_data/
│   ├── extracted_features.pkl                    # Extracted features (749 windows)
│   └── hmm_trained_model.pkl                     # Trained HMM model
└── README.md
```

## Key Features

- **Sensor Fusion**: Merges accelerometer and gyroscope data from smartphone sensors
- **Window-based Processing**: Overlapping temporal windows (2-second windows with 1-second stride)
- **Comprehensive Feature Engineering**: 
  - **Time-domain**: Mean, std, variance, min/max, median, correlations, signal magnitude area
  - **Frequency-domain**: Dominant frequency, spectral energy (via FFT)
- **Sequence Modeling**: Gaussian HMM with diagonal covariance captures temporal patterns
- **Robust Evaluation**: Recording-based train/test split prevents data leakage

## Dataset

**Source**: Smartphone accelerometer and gyroscope recordings  
**Sampling Rate**: 50 Hz  
**Activities**: 4 classes
- **Jump**: 11 recordings (~1.47 minutes)
- **Standing**: 16 recordings (~2.28 minutes)
- **Still**: 11 recordings (~1.53 minutes)  
- **Walk**: 12 recordings (~1.56 minutes)

**Total**: 50 recordings, ~410 seconds (6.84 minutes) of sensor data

## Methodology

### 1. Data Preprocessing
- Normalize timestamps and axis columns
- Merge accelerometer and gyroscope data using nearest-neighbor temporal alignment
- Handle missing values and duplicate timestamps
- Standardize to 50 Hz sampling rate

### 2. Feature Extraction
- Segment signals into overlapping windows (100 samples, 50-sample stride)
- Compute 52 features per window across 6 axes (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
  - **Time-domain**: Mean, std, variance, min/max, median, correlations, SMA
  - **Frequency-domain**: Dominant frequency, spectral energy
- Result: 749 feature windows from 50 recordings

### 3. Model Training
- **Algorithm**: Gaussian HMM with diagonal covariance
- **Training**: Baum–Welch (EM algorithm) for parameter estimation
- **States**: 4 hidden states (one per activity)
- **Normalization**: Z-score scaling to handle heterogeneous sensor units
- **Convergence**: Achieved in 14 iterations

### 4. Sequence Decoding
- **Algorithm**: Viterbi algorithm for finding most likely activity sequence
- **State-to-Activity Mapping**: Majority vote on training data

## Results

**Test Accuracy**: **98.7%** (157 windows, 10 held-out recordings)

### Classification Report

| Activity  | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Jump      | 1.000     | 1.000  | 1.000    | 61      |
| Standing  | 1.000     | 0.957  | 0.978    | 46      |
| Still     | 1.000     | 1.000  | 1.000    | 35      |
| Walk      | 0.882     | 1.000  | 0.938    | 15      |
| **Macro Avg** | 0.971 | 0.989 | 0.979 | 157     |

**Key Observations**:
- Perfect classification for jump, still, and most walk instances
- Minor confusion: 2 standing samples misclassified as walk
- All activities show distinct sensor signatures enabling robust separation

### Model Insights
- **Jump**: Large spikes in both accelerometer and gyroscope signals
- **Walk**: Smooth, repeating rhythmic patterns
- **Standing**: Small initial movements settling into near-stationary state
- **Still**: Flat signals with minimal variation

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages (see [Requirements](#requirements))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Hidden-Markov-Model.git
   cd Hidden-Markov-Model
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scipy scikit-learn hmmlearn matplotlib seaborn joblib
   ```

3. **Run the notebooks**
   ```bash
   jupyter notebook notebooks/
   ```

### Usage

Execute notebooks in order:
1. `data_collection_and_preprocessing.ipynb` - Process raw sensor data
2. `feature-extraction.ipynb` - Extract features from merged data
3. `hmm-model.ipynb` - Train and evaluate the HMM

**Note**: If using Google Colab (as in original), update file paths accordingly.

## Notebooks Guide

### 1. Data Collection and Preprocessing
- Locates accelerometer and gyroscope CSV files
- Merges sensor data by timestamp
- Visualizes raw signals for each activity
- Outputs: Merged CSV files with activity labels

### 2. Feature Extraction  
- Applies sliding window segmentation
- Computes 52 time- and frequency-domain features
- Saves processed features to `processed_data/extracted_features.pkl`

### 3. HMM Model
- Splits data into train/test sets (by recording)
- Scales features with StandardScaler
- Trains Gaussian HMM with Baum–Welch
- Evaluates using Viterbi decoding
- Visualizes confusion matrix, transition matrix, and activity sequences
- Saves model to `processed_data/hmm_trained_model.pkl`

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
hmmlearn>=0.2.7
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
jupyter>=1.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Future Improvements

- [ ] Add cross-validation to assess generalization
- [ ] Experiment with different HMM configurations (Gaussian mixtures, covariance types)
- [ ] Implement real-time activity recognition on streaming sensor data
- [ ] Extend to additional activities (running, sitting, etc.)
- [ ] Deploy as a mobile app for on-device inference
- [ ] Add confidence scores and uncertainty quantification
- [ ] Compare HMM performance with deep learning approaches (LSTM, Transformer)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Developed as part of a Machine Learning summative project focusing on temporal sequence modeling and human activity recognition.

## Acknowledgments

- Datasets collected using smartphone sensors
- `hmmlearn` library for HMM implementation
- Scientific Python ecosystem for data processing and visualization

---

**Keywords**: Hidden Markov Models, Activity Recognition, Sensor Fusion, Time Series Classification, Probabilistic Models, Baum–Welch, Viterbi Algorithm, Machine Learning
