import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Global variable for the dataset
data = None

def load_dataset():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path, sep=None, engine="python")
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            update_column_dropdowns()
            display_dataset()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

def display_dataset():
    if data is not None:
        # Display first 5 rows of the dataset
        print("\n--- Dataset (First 5 Rows) ---\n")
        print(data.head())
        print("\n--- Summary Statistics ---\n")
        print(data.describe())
    else:
        messagebox.showinfo("Info", "No dataset loaded.")

def show_null_values():
    if data is not None:
        print("\n--- Null Values ---\n")
        print(data.isnull().sum())
    else:
        messagebox.showinfo("Info", "No dataset loaded.")

def plot_correlation_heatmap():
    if data is not None:
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=["int64", "float64"])
        if not numeric_data.empty:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()
        else:
            messagebox.showinfo("Info", "Dataset does not contain any numeric columns for correlation analysis.")
    else:
        messagebox.showinfo("Info", "No dataset loaded.")

def plot_histogram():
    if data is not None:
        column = column_dropdown.get()
        if column in data.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(data[column]):
                plt.figure(figsize=(8, 5))
                sns.histplot(data[column], kde=True)
                plt.title(f"Histogram of {column}")
                plt.show()
            else:
                messagebox.showerror("Error", f"Column '{column}' is not numeric and cannot be plotted.")
        else:
            messagebox.showerror("Error", "Column not found in dataset.")
    else:
        messagebox.showinfo("Info", "No dataset loaded.")

def train_linear_regression():
    if data is not None:
        target = target_dropdown.get()
        if target in data.columns:
            # Ensure target column is numeric
            if not pd.api.types.is_numeric_dtype(data[target]):
                messagebox.showerror("Error", f"Target column '{target}' is not numeric.")
                return

            # Select only numeric features, excluding the target column
            X = data.select_dtypes(include=["int64", "float64"]).drop(columns=[target])
            y = data[target]

            try:
                # Split the dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Test the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Display metrics
                print("\n--- Model Metrics ---\n")
                print(f"Mean Squared Error: {mse}")
                print(f"R2 Score: {r2}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model: {e}")
        else:
            messagebox.showerror("Error", "Target column not found in dataset.")
    else:
        messagebox.showinfo("Info", "No dataset loaded.")

def update_column_dropdowns():
    if data is not None:
        # Select only numeric columns
        numeric_columns = list(data.select_dtypes(include=["int64", "float64"]).columns)

        # Update dropdowns
        column_dropdown["values"] = numeric_columns
        target_dropdown["values"] = numeric_columns

# GUI
root = tk.Tk()
root.title("Data Analysis Tool")
root.geometry("500x400")

tk.Button(root, text="Load Dataset", command=load_dataset).pack(pady=5)
tk.Button(root, text="Display Dataset", command=display_dataset).pack(pady=5)
tk.Button(root, text="Show Null Values", command=show_null_values).pack(pady=5)
tk.Button(root, text="Plot Correlation Heatmap", command=plot_correlation_heatmap).pack(pady=5)

# Visualization
tk.Label(root, text="Select column for histogram:").pack()
column_dropdown = ttk.Combobox(root, state="readonly")
column_dropdown.pack()
tk.Button(root, text="Plot Histogram", command=plot_histogram).pack(pady=5)

# Machine Learning
tk.Label(root, text="Select target column for Linear Regression:").pack()
target_dropdown = ttk.Combobox(root, state="readonly")
target_dropdown.pack()
tk.Button(root, text="Train Linear Regression", command=train_linear_regression).pack(pady=5)

root.mainloop()
