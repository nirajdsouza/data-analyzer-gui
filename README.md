# Data Analysis Tool

## Overview
A Python-based GUI application for data analysis and visualization using `tkinter`. Load a CSV file, explore the dataset, visualize key insights, and train a linear regression model.

## Features
- **Load Dataset**: Upload a CSV file.
- **View Data**: Display the first 5 rows and basic statistics.
- **Analyze Null Values**: Identify missing data.
- **Visualizations**:
  - Correlation Heatmap for numeric features.
  - Histogram for a selected column.
- **Train Linear Regression**: Build and evaluate a model with metrics like MSE and R².

## Prerequisites
- Python 3.6+
- Install dependencies:
  ```bash
  pip install pandas matplotlib seaborn scikit-learn
  ```
  
## Usage

1.  Run the script:
    ```bash
    python script_name.py
    ```
    
2.  Use the GUI to load and analyze your dataset, create visualizations, or train a model.

## Example Datasets

You can test the program with the following datasets:

1.  Wine Quality Dataset: [Download](https://archive.ics.uci.edu/dataset/186/wine+quality)
2.  Air Quality Data: [Download](https://archive.ics.uci.edu/dataset/360/air+quality)
3.  Iris Dataset: [Download](https://archive.ics.uci.edu/ml/datasets/Iris)

Ensure that the files used contain numerical data and are in CSV format.

## Notes

-   Only numeric columns are used for heatmaps, histograms, and modeling.
-   Supports CSV files only.

## License

[MIT](https://github.com/nirajdsouza/data-analyzer-gui/blob/main/LICENSE) License.