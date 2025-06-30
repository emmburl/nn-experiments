"""
For each variable of interest, plot the minimum test loss over all combinations of other variables
as a function of quantization precision (best case scenario).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the CSV file
df = pd.read_csv('C:/Users/ejb13/Downloads/experiment_results_largerange.csv')

x_col = 'quantizationPrecision'
y_col = 'testLoss'
variables_of_interest = ['dataset', 'numLayers', 'numNodes', 'featureSet', 'numFeatures', 'quantizationMethod']

log_y = True  # Set to False for linear y-axis
log_y_base = 2  # Use 2 for more detail at the low end, or 10 for standard log

for variable_of_interest in variables_of_interest:
    # Group by quantizationPrecision and the variable of interest, then take minimum testLoss
    grouped = df.groupby([x_col, variable_of_interest])[y_col].min().reset_index()

    plt.figure(figsize=(8, 5))
    for value in grouped[variable_of_interest].unique():
        subset = grouped[grouped[variable_of_interest] == value]
        plt.plot(subset[x_col], subset[y_col], marker='o', label=f"{variable_of_interest}={value}")

    plt.xlabel(x_col)
    plt.ylabel(f"Minimum {y_col}")
    plt.title(f"Minimum {y_col} vs {x_col} for each {variable_of_interest} (Best Case)")
    plt.xscale('log', base=2)

    # Set x-ticks to actual values
    x_vals = sorted(df[x_col].unique())
    plt.xticks(x_vals, labels=[str(v) for v in x_vals])

    if log_y:
        plt.yscale('log', base=log_y_base)
        plt.ylabel(f"Minimum {y_col} (log base {log_y_base})")

    plt.legend(title=variable_of_interest)
    plt.tight_layout()
    plt.savefig(f'nn_experiment_result_plot_minima_{variable_of_interest}_largerange.pdf', bbox_inches='tight')
    plt.close() 