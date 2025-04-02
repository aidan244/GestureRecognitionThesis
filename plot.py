import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the CSV file name
csv_filename = '/Users/aidantang/Desktop/Gesture_recog/Classifiers/results_comparison.csv'

# Check if the file exists
if not os.path.isfile(csv_filename):
    print(f"Error: {csv_filename} not found. Please generate the results first.")
else:
    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Ensure PatientID and ForearmPercentage are numeric
    df['PatientID'] = pd.to_numeric(df['PatientID'], errors='coerce')
    df['ForearmPercentage'] = pd.to_numeric(df['ForearmPercentage'], errors='coerce')
    df['TestAccuracy'] = pd.to_numeric(df['TestAccuracy'], errors='coerce')
    
    # Compute and print average accuracy for each model
    avg_accuracies = df.groupby('Model')['TestAccuracy'].mean()
    print("\nAverage Test Accuracies by Model:")
    print(avg_accuracies)

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=80)  # Lower DPI

    # --- Plot 1: Accuracy vs PatientID ---
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        axes[0].plot(model_data['PatientID'], model_data['TestAccuracy'], marker='o', linestyle='-', label=model)

    # Formatting for first plot
    axes[0].set_xlabel('Patient ID')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Model Accuracy Comparison Across Patients')
    axes[0].legend(title='Model')
    axes[0].grid(True)

    # --- Plot 2: Accuracy vs Forearm Percentage (Scatter + Best Fit Line) ---
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        axes[1].scatter(model_data['ForearmPercentage'], model_data['TestAccuracy'])
        
        # Compute line of best fit if there are at least two points
        if len(model_data) > 1:
            try:
                coeffs = np.polyfit(model_data['ForearmPercentage'], model_data['TestAccuracy'], deg=1)
                axes[1].plot(model_data['ForearmPercentage'], np.polyval(coeffs, model_data['ForearmPercentage']), linestyle='--', label=f'{model} Trend')
            except np.linalg.LinAlgError:
                print(f"Skipping model {model} due to fitting error.")

    # Formatting for second plot
    axes[1].set_xlabel('Forearm Percentage')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Model Accuracy vs Forearm Percentage')
    axes[1].legend(title='Model')
    axes[1].grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
