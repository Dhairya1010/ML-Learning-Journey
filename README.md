# Decision Tree vs Random Forest Regression

This project compares the performance of **Decision Tree Regressor** and **Random Forest Regressor** on a housing dataset using Python and scikit-learn.

## Objective
To explore how ensemble learning (Random Forest) improves prediction accuracy compared to a single Decision Tree.

## Dataset
The dataset contains housing data, including median house value, housing age, total bedrooms, and total rooms. Missing values have been removed.

## Features Used
- housing_median_age
- total_bedrooms
- total_rooms

## Models
- DecisionTreeRegressor (max_leaf_nodes=125)
- RandomForestRegressor (max_leaf_nodes=125)

## Evaluation Metric
- Mean Absolute Error (MAE)

## What I Learned
- How to split data into training and validation sets
- How model parameters affect performance
- The power of ensemble methods like Random Forest
