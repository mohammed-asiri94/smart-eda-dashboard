# Smart EDA Dashboard

Smart EDA Dashboard is an interactive web application built with Python and Streamlit to simplify exploratory data analysis for CSV and Excel files.

## Project Overview

The goal of this project is to make the initial data analysis workflow faster and more organized. Users can upload a dataset, explore data quality, detect missing values, identify duplicates and outliers, generate visualizations, apply basic cleaning steps, and run selected statistical models.

## Features

- Upload CSV and Excel files
- Display dataset overview and summary statistics
- Detect missing values and duplicate rows
- Identify possible ID columns and high-cardinality variables
- Detect outliers using the IQR method
- Generate smart data quality recommendations
- Apply data cleaning and imputation methods
- Create interactive visualizations
- Analyze correlations between numeric variables
- Run statistical models:
  - Linear Regression
  - Binary Logistic Regression
  - Poisson Regression
  - Negative Binomial Regression
- Download cleaned datasets and reports

## Tools Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- statsmodels

## How to Run Locally

Install the required packages:

```bash
pip install -r requirements.txt