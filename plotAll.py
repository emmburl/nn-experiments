from random import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file
df = pd.read_csv('C:/Users/ejb13/Downloads/experiment_results(22).csv')

# Identify the columns
x_col = 'quantizationPrecision'
y_col = 'testLoss'
group_cols = [col for col in df.columns if col not in [x_col, y_col]]

# Group by all other columns
print(f"Number of curves: {len(list(df.groupby(group_cols)))}") # prints number of curves plotted

print("Columns:", group_cols)
for col in group_cols:
    print(f"{col}: {df[col].unique()}")

print("Number of rows in CSV:", len(df))
print("Number of unique combinations:", len(df.drop_duplicates(subset=group_cols)))

for group_values, group_df in df.groupby(group_cols):
    label = ', '.join(f'{col}={val}' for col, val in zip(group_cols, group_values))
    plt.plot(group_df[x_col], group_df[y_col], marker='o', label=label)

plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title('Test Loss vs Quantization Precision for Different Network Configurations')
plt.xscale('log', base=2)  # Set x-axis to log base 2
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=3)
plt.tight_layout(rect=[0, 0.15, 1, 1])  # Make room at the bottom
plt.savefig('nn_experiment_result_plot.pdf', bbox_inches='tight')  # Save the plot to a PDF
plt.show()

