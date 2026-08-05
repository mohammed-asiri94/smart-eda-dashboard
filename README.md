# Smart EDA Dashboard

Smart EDA Dashboard is an interactive web application built with Python and Streamlit to support exploratory data analysis, data quality assessment, data cleaning, visualization, and statistical modeling for CSV and Excel files.

## Live Demo

Try the app here:

https://smart-eda-dashboard.streamlit.app

## Project Overview

The goal of this project is to make the early stages of data analysis faster, more organized, and easier to perform through an interactive web interface.

When working with a new dataset, analysts usually need to understand the structure of the data, check data quality, identify missing values, detect duplicates and outliers, explore relationships between variables, and sometimes run basic statistical models. This project brings these steps into one dashboard instead of relying on separate scripts for each task.

The project is still an initial version and is being continuously improved. Future updates may include better model guidance, additional diagnostics, enhanced user experience, and more advanced analytical options.

## Key Features

* Upload CSV and Excel files directly from the browser
* Display dataset overview and summary statistics
* Preview raw data and column information
* Detect missing values and missing percentages
* Identify duplicate rows
* Detect possible ID columns
* Identify high-cardinality categorical variables
* Detect constant columns
* Detect date-like columns
* Detect outliers using IQR and Z-Score methods
* Generate smart data quality recommendations
* Apply data cleaning steps
* Remove duplicate rows
* Remove constant columns
* Remove possible ID columns
* Drop columns based on missing-value percentage
* Apply numeric and categorical imputation methods
* Apply group-based imputation
* Apply KNN imputation
* Apply iterative imputation
* Generate multiple imputed datasets
* Cap or remove outliers using IQR or Z-Score
* Download cleaned datasets
* Generate interactive charts
* Analyze correlations between numeric variables
* Run statistical tests
* Generate downloadable reports
* Export reports as CSV, HTML, and Excel files
* Normalize uploaded tabular formats to temporary Parquet storage
* Use exact DuckDB queries for core metrics in Large Data Mode (100,000+ rows)
* Run expensive quality checks, outlier detection, summaries, and profiling only when requested
* Reuse completed analysis results across Streamlit reruns until the dataset or relevant settings change
* Calculate histogram, ECDF, density, categorical, and time-bucket aggregates from all rows
* Limit browser-heavy point charts to a deterministic, disclosed display-only sample
* Keep multivariate point charts to 5,000 displayed observations without sampling model inputs
* Show a row-accounting audit for every model family, including source, eligible, excluded, training, and test rows
* Never apply display sampling to statistical or machine-learning model inputs
* Start machine-learning training only after an explicit Run action
* Stop models by default when any row would be excluded; temporary model-only exclusion requires explicit user permission

## Visualization Features

The dashboard includes several visualization options for different types of analysis.

### Univariate Analysis

* Histogram with distribution view
* Boxplot
* Violin plot
* ECDF plot
* Strip plot
* Bar chart
* Horizontal bar chart
* Pie chart
* Treemap
* Funnel chart

### Bivariate Analysis

* Scatter plot with trendline
* Density heatmap
* 2D density contour
* Bubble chart
* Grouped boxplot
* Grouped violin plot
* Strip plot by category
* Mean ± SD bar chart
* Categorical heatmap
* Grouped and stacked bar charts

### Multivariate Analysis

* Pair plot
* Scatter matrix
* Parallel coordinates plot
* Radar chart
* 3D scatter plot

### Time and Advanced Visualizations

* Line chart
* Area chart
* Sunburst chart
* Waterfall chart
* Calendar-style heatmap

## Statistical Modeling

The dashboard includes several statistical modeling sections.

### Base Models

* Linear Regression
* Binary Logistic Regression
* Poisson Regression
* Negative Binomial Regression

### Survival Analysis

* Kaplan-Meier survival analysis
* Cox Proportional Hazards model

### Time Series Analysis

* Time series visualization
* Decomposition into trend, seasonality, and residuals
* Stationarity tests using ADF and KPSS
* ACF and PACF plots
* ARIMA forecasting
* Forecast table download

### Interrupted Time Series

* Intervention point selection
* Pre- and post-intervention comparison
* Level change estimation
* Slope change estimation
* Counterfactual trend visualization
* Durbin-Watson residual diagnostic
* Chow test for structural break
* Optional controlled ITS using a control series

### Mixed Effects Models

* Linear mixed effects modeling
* Group-level structure support
* Random effects analysis

### Causal Inference

* Propensity Score Matching
* Inverse Probability Weighting
* Treatment effect estimation

## Reports and Downloads

The app allows users to download different outputs, including:

* Cleaned dataset as CSV
* Missing values report
* Column type report
* Unique values report
* Outlier report
* Smart recommendations
* Full HTML report
* Excel report with multiple sheets
* Model summaries and outputs
* Forecast tables
* Multiple imputed datasets as ZIP files

## Tools and Libraries Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* scikit-learn
* statsmodels
* SciPy
* openpyxl
* xlrd
* pyarrow
* DuckDB

## Project Structure

```text
smart-eda-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── run_app.bat
├── .gitignore
├── .streamlit/
├── services/
│   ├── __init__.py
│   ├── error_service.py
│   ├── dataset_service.py
│   ├── file_service.py
│   ├── quality_service.py
│   ├── session_service.py
│   ├── upload_controller.py
│   └── visual_service.py
│
├── ui/
│   ├── __init__.py
│   ├── cleaning_page.py
│   ├── correlation_page.py
│   ├── layout.py
│   ├── model_page.py
│   ├── overview_page.py
│   ├── quality_page.py
│   ├── raw_data_page.py
│   ├── reports_page.py
│   ├── sidebar.py
│   └── visual_page.py
│
└── modules/
    ├── __init__.py
    ├── data_quality.py
    ├── cleaning.py
    ├── reports.py
    ├── statistical_tests.py
    ├── data_profiling.py
    ├── large_data_engine.py
    │
    └── models/
        ├── __init__.py
        ├── base_models.py
        ├── survival.py
        ├── time_series.py
        ├── mixed_effects.py
        ├── data_audit.py
        └── causal.py
```

## Automated Tests

Run the permanent regression suite from the project directory:

```bash
python -m unittest discover -s tests -v
```

The suite verifies file validation and session invalidation, Parquet-backed
ingestion, preservation of all 500,000 rows, absence of model sampling, and
the requirement for an explicit user action before cleaning.

## How to Run Locally

Install Python 3.11 or 3.12 and enable **Add Python to PATH** during setup.
On Windows, `run_app.bat` now detects either `python` or the `py -3` launcher,
installs missing project dependencies, and starts Streamlit. If Python itself
is missing, it displays installation instructions instead of closing silently.

Clone the repository:

```bash
git clone https://github.com/mohammed-asiri94/smart-eda-dashboard.git
```

Move into the project folder:

```bash
cd smart-eda-dashboard
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

## Notes

* The app is designed for exploratory and educational analysis.
* Results should be interpreted carefully, especially statistical models and causal inference outputs.
* Users should avoid uploading sensitive or confidential data to the public web version.
* Pickle files (`.pkl` and `.pickle`) are intentionally unsupported because loading them can execute untrusted code.
* Uploaded files are validated before parsing. The public app currently enforces a 100 MB upload safety limit.
* Files with 100,000 rows or more use disk-backed Parquet storage. Core metrics and models continue to use all eligible rows; any future visual sampling must be disclosed separately.
* Temporary normalized datasets are removed after their retention window and are not written into the project repository.
* The project is still under active development and may continue to change as new features and improvements are added.

## Future Improvements

Planned improvements may include:

* Better automated model guidance
* More detailed model diagnostics
* Enhanced error handling
* Improved mobile layout
* More visualization customization
* Additional statistical tests
* More advanced time series options
* Better documentation and examples

## Author

Developed by Mohammed Asiri.

## Links

Live App:

https://smart-eda-dashboard.streamlit.app

GitHub Repository:

https://github.com/mohammed-asiri94/smart-eda-dashboard
