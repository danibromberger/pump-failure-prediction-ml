# Pump Failure Prediction Project

This is the codebase for the paper titled *"INTERPRETABLE MACHINE LEARNING IN INDUSTRIAL WATER PUMP FAILURE PREDICTION: THE ROLE OF SHAPLEY ADDITIVE EXPLANATIONS VALUES AND DATA SAMPLING TECHNIQUES"*, submitted for publication.

## 📁 Repository Structure

- `/data/`: Dataset used in the experiments (`sensor.csv`) Dataset download link: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data
- `/scripts/`: All Python scripts used in preprocessing, modeling, and evaluation
  - `preprocessing.py`: Data cleaning, encoding, and balancing
  - `pca_analysis.py`: Dimensionality reduction using Principal Component Analysis
  - `model_training.py`: Model training with Random Forest and hyperparameter tuning
  - `shap_analysis.py`: Interpretability using SHAP values

## ⚙️ Requirements

To run the code, you'll need:

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, shap
