# Exploratory Data Analysis and Machine Learning Modeling (TOW Prediction)

## Project Description

The goal of this project is to perform exploratory data analysis (EDA) and build machine learning models to predict the **Actual Takeoff Weight (TOW)** based on available flight data.

The project includes:

- In-depth analysis of distributions, correlations, and outliers,
- Investigation of missing data and its impact on modeling,
- Exploration of geographic and temporal dependencies,
- Implementation and evaluation of various regression models.

## Project Structure

/project-root
│
├── datasets/ # Source data
│ EDA.ipynb # Jupyter notebooks for analysis and modeling
│ ML.ipynb
├── utils/ # Scripts for preprocessing and model training
│ ├── constants.py
│ └── utils.py
├── requirements.txt # Python package dependencies
└── README.md # This file

## Technologies and Libraries

- Python 3.8+
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

2. Perform exploratory data analysis by running notebooks/eda.ipynb.

3. Train regression models using notebooks/modeling.ipynb or scripts/model_training.py.

For questions, please contact:
Kacper Samulski
samulski04@gmail.com
