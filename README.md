# Abstract
This study aims to reproduce the results of the paper “In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis” by Nonaka and Seita (2021), which benchmarks several deep learning models for ECG classification. The team reproduced results using two architectures—ResNet1D and NF-ResNet1D—on the G12EC dataset using three different loss functions: Binary Cross-Entropy (BCE), Focal Loss, and Asymmetric Loss. Our results affirm the paper’s claim that ResNet-18 serves as a strong baseline, with no consistently superior architecture across configurations. The team also explored model performance under different loss functions. Results show that Asymmetric Loss led to the highest F1 score, suggesting it is better suited for imbalanced ECG data



# Enviroment
- Python 3.9.2

- torch 2.4.1

- numpy 1.24.2

- matplotlib 3.9.4

- pyhealth 1.1.6

- scikit-learn 1.5.2

- tqdm 4.65.0

# Reproduction Steps

1. Download data from this [link](https://www.kaggle.com/bjoernjostein/georgia-12lead-ecg-challenge-database/metadata)
2. Unzip in ./data
3. Rename  directory to ./data/WFDB
4. Run preprocessing

  `python dataprep.py`

6. Run model

 `python ecg.py`
