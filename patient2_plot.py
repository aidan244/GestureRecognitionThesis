import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data (adjust the file path if necessary)
df = pd.read_csv('Classifiers/results_comparison.csv')

# Group by model to get average accuracy and standard deviation across ALL patients
df_agg = df.groupby('Model')['TestAccuracy'].agg(['mean', 'std']).reset_index()

# Create a bar chart with error bars
plt.figure(figsize=(10, 6))

# x positions for each model
x = range(len(df_agg))

# Plot bar chart: 'mean' as height, 'std' as the error bar
plt.bar(x, df_agg['mean'], yerr=df_agg['std'], capsize=5, color='skyblue')

# Customize plot
plt.title("Average Model Accuracy Across All Patients")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # If accuracy is in [0,1]. Adjust if needed.

# Use model names as x-tick labels
plt.xticks(x, df_agg['Model'], rotation=45)

plt.tight_layout()
plt.show()
