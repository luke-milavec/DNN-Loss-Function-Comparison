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
