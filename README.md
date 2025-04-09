# **Diamond Price Prediction Project**

## **Project Overview**
This project focuses on analyzing and predicting diamond prices using Feedforward Neural Network implemented from scratch. By leveraging key features of diamonds, such as `carat`, `x`, `y`, and `z`, we build a predictive model to estimate prices. The project includes:
1. **Exploratory Data Analysis (EDA)**: Understanding the dataset, visualizing trends, and selecting relevant features.
2. **Model Training**: Developing a Feedforward Neural Network to predict diamond prices based on selected features.

---

## **Dataset**
- **Source**: [Diamonds Dataset on Kaggle](https://www.kaggle.com/datasets/shivam2503/diamonds)
- **Description**: The dataset contains detailed information about diamonds, including their characteristics (`carat`, `cut`, `color`, `clarity`, `x`, `y`, `z`) and prices.

---

## **Project Components**

### **1. Notebooks**
- **EDA Notebook (`eda_diamonds.ipynb`)**:
  - Explores the dataset to gain insights into feature distributions, correlations, and outliers.
  - Includes justifications for feature selection (`carat`, `x`, `y`, `z`) based on their strong positive correlation with `price`.
  - Visualizes outliers to ensure data integrity.

- **Model Training Notebook (`model_training.ipynb`)**:
  - Preprocesses the data by splitting it into training and test sets and applying log transformation to `price` for normalization.
  - Trains a neural network model with selected features and evaluates performance.
  - Includes visualizations like the loss curve, residual plot, and predictions vs actual values.

### **2. Scripts**
- **`data_processing.py`**:
  - Automates data preprocessing tasks such as filtering relevant columns and splitting data into training and testing sets.
  - Saves processed datasets to structured directories for reproducibility.

- **`model_training.py`**:
  - Defines the `DiamondPricePredictor` class, a neural network model with a customizable architecture for training and predicting prices.
  - Includes methods for forward propagation, backpropagation, and saving the model.

---

## **Key Features**

### **1. Feature Selection**
- Based on the EDA results, numerical features with strong positive correlations to `price` (`carat`, `x`, `y`, `z`) were selected.
- Categorical features (`cut`, `color`, `clarity`) were excluded due to their weak linear correlation with `price`.

### **2. Model Performance**
- The model achieved an impressive **Mean Squared Error (MSE)** of `0.0640` on the test set.
- Outperformed the baseline MSE of `0.9556` by over 93%, demonstrating its effectiveness in predicting diamond prices.

### **3. Visualizations**
- **EDA**:
  - Histograms and bar plots to understand feature distributions.
  - Correlation matrix to identify relationships between variables.
  - Outlier detection with box plots and scatter plots.
- **Model Analysis**:
  - Loss curve to monitor model convergence.
  - Residual plot to verify prediction quality.
  - Predictions vs actual values to validate model performance.

---
## **Next Steps**
Although this project is already functional and demonstrates solid results, future improvements could include:
1. **Incorporating Categorical Features**: Encode `cut`, `color`, and `clarity` to capture their non-linear influence on price.
2. **Hyperparameter Tuning**: Experiment with different model architectures, learning rates, and batch sizes to optimize performance.
3. **Deployment**: Develop a web interface or API to make the model accessible for real-time predictions.

---

## **How to Use**

### **1. Prerequisites**
Ensure the following libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

Install additional dependencies as needed:
```bash
pip install -r requirements.txt

