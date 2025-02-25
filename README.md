# Temperature and Housing Price Prediction with PyTorch

This repository contains code to predict **temperatures** (Celsius to Fahrenheit) and **housing prices** using both **linear** and **nonlinear models** built with **PyTorch**.

The project includes three different scripts:
1. **Comparing Nonlinear and Linear Models for Temperature Prediction**  
2. **Predicting Housing Prices Using Linear Regression with PyTorch**  
3. **Predicting Housing Prices Using Linear Regression with PyTorch: A Feature-Driven Approach**

## Table of Contents
- [1. Temperature Prediction](#1-temperature-prediction)
- [2. Housing Price Prediction (Full Features)](#2-housing-price-prediction-full-features)
- [3. Housing Price Prediction (Feature-Driven Approach)](#3-housing-price-prediction-feature-driven-approach)
- [4. How to Use](#4-how-to-use)
- [5. Dependencies](#5-dependencies)
- [6. License](#6-license)

---

## 1. Temperature Prediction

This script compares **linear** and **nonlinear** models to predict temperature. We use **Celsius to Fahrenheit** conversion data and apply both types of models:
- The **nonlinear model** uses a quadratic equation.
- The **linear model** uses a simple linear relationship.

### Key Features:
- Compares two models (linear vs. nonlinear).
- Trains the models with different learning rates.
- Plots predictions versus actual data.

---

## 2. Housing Price Prediction (Full Features)

This script uses a **linear regression model** to predict housing prices based on multiple features such as:
- Area, bedrooms, bathrooms, parking, and others.
- Categorical variables (one-hot encoded) like `mainroad`, `guestroom`, and `airconditioning`.

### Key Features:
- Preprocessing with **one-hot encoding** for categorical data.
- **Standardization** of numerical features.
- Trains a linear regression model using **PyTorch**.
- Evaluates model performance on training and validation sets.

---

## 3. Housing Price Prediction (Feature-Driven Approach)

This script is a focused version of the previous housing price prediction model. Instead of using all features, this model only uses a subset of features such as:
- Area, bedrooms, bathrooms, stories, and parking.

### Key Features:
- Focuses on a smaller set of features to predict housing prices.
- Trains and evaluates a linear regression model using **PyTorch**.
- Visualizes predictions against actual prices.

---

## 4. How to Use

  ### Step 1: Clone the Repository
To clone this repository, use the following command:
bash
git clone https://github.com/akafle01/temperature-and-housing-price-predictions-pytorch.git


  ###Step 2: Install Dependencies
This project requires PyTorch, scikit-learn, pandas, and matplotlib. Install them using pip:
pip install torch scikit-learn pandas matplotlib

  ###Step 3: Run the Code
Each script can be run independently. Here are the scripts to run:

Temperature Prediction:
Run temperature_model.py to train and compare the nonlinear and linear models for temperature prediction.

Housing Price Prediction (Full Features):
Run housing_price_model.py to train the linear regression model on all available features.

Housing Price Prediction (Feature-Driven Approach):
Run feature_driven_housing_model.py to predict housing prices using a reduced feature set.

Datasets:
This repository uses a housing dataset (Housing (1).csv). It contains various features such as the area, number of bedrooms, and bathrooms. The target variable is the price of the house.

---

## 5. Dependencies
Python 3.7+
PyTorch
scikit-learn
pandas
matplotlib
Install the required dependencies by running:

pip install torch scikit-learn pandas matplotlib

---

## 6. License
This project is licensed under the MIT License - see the LICENSE file for details.

---

### What this README includes:

1. **Project Overview**: Briefly describes the project and what each script does.
2. **How to Use**: A step-by-step guide to set up and run your code. This includes instructions for cloning the repository, installing dependencies, and running each script.
3. **Dataset Information**: Mentions the dataset you're using (`Housing (1).csv`).
4. **Dependencies**: Lists all the libraries required for the project and how to install them.
5. **License**: Mentions that the project is licensed under the MIT License. If you're using a different license, you can adjust this.

