import numpy as np
import pandas as pd
from scipy.io import loadmat
from matplotlib import pyplot as plt

def plot(df, metric, normalize=False, filename="plot.png"):
    # Clean DataFrame by ensuring all entries are scalars
    df_cleaned_final = df.applymap(lambda x: x if not isinstance(x, np.ndarray) else x[0] if x.size == 1 else x)
    df_cleaned_final['N_t'] = pd.to_numeric(df_cleaned_final['N_t'], errors='coerce')
    df_cleaned_final['N_x'] = pd.to_numeric(df_cleaned_final['N_x'], errors='coerce')
    df_cleaned_final[metric] = pd.to_numeric(df_cleaned_final[metric], errors='coerce')
    
    # Apply normalization if specified
    if normalize:
        df_cleaned_final[metric] = df_cleaned_final[metric] / df_cleaned_final['Std']
        ylabel = f'Normalized {metric}'
        title = f'N_t vs. Normalized {metric} for Different N_x Values'
    else:
        ylabel = metric
        title = f'N_t vs. {metric} for Different N_x Values'

    # Now plot N_t vs. metric with each line representing an N_x value in specified colors
    plt.figure(figsize=(10, 6))
    colors = {10: 'red', 30: 'green', 50: 'blue'}

    # Group by N_x and plot each group with specified colors
    for Nx_value, group in df_cleaned_final.groupby('N_x'):
        if Nx_value in colors:  # Plot only the specified N_x values
            plt.plot(group['N_t'], group[metric], label=f'N_x = {Nx_value}', color=colors[Nx_value], marker='o')

    # Adding labels and title
    plt.xlabel('N_t')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='N_x')
    plt.xscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save plot to file
    plt.savefig(filename)
    plt.close()  # Close the plot to prevent displaying it

# Load .mat file
mat_data = loadmat('simulation_results.mat')

# Flatten the data
raw_results = mat_data['results'].flatten()

# Initialize lists to store data
Nt_list = []
Nx_list = []
RMSE_list = []
Corr_list = []
Time_list = []
StdDev_list = []

# Iterate through entries, extracting non-empty arrays
for entry in raw_results:
    if entry.size > 0:  # Check for non-empty entries
        Nt = entry[0][0][0][0]  # Extract N_t scalar value
        Nx = entry[0][0][1][0]  # Extract N_x scalar value
        RMSE = entry[0][0][2][0][0]  # Extract RMSE scalar value
        Corr = entry[0][0][3][0][0]  # Extract Corr scalar value
        Std = entry[0][0][4][0][0]  # Extract Std scalar value
        computation_time = entry[0][0][5][0][0]  # Extract computation time scalar value
        
        # Append to lists
        Nt_list.append(Nt)
        Nx_list.append(Nx)
        RMSE_list.append(RMSE)
        Corr_list.append(Corr)
        StdDev_list.append(Std)
        Time_list.append(computation_time)

# Create DataFrame directly from lists with updated column names
df = pd.DataFrame({
    'N_t': Nt_list,
    'N_x': Nx_list,
    'RMSE': RMSE_list,
    'Corr': Corr_list,
    'Std': StdDev_list,
    'computation_time': Time_list
})
df['N_t'] = df['N_t'].astype(int)
df['N_x'] = df['N_x'].astype(int)

# Save the normalized RMSE plot
plot(df, 'RMSE', normalize=True, filename="normalized_rmse_plot.png")
plot(df, 'Corr', normalize=False, filename="Corr_plot.png")

# Save computation time plot across N_t for different N_x values
plt.figure(figsize=(10, 6))
colors = {10: 'red', 30: 'green', 50: 'blue'}
for Nx_value, group in df.groupby('N_x'):
    if Nx_value in colors:  # Plot only the specified N_x values
        plt.plot(group['N_t'], group['computation_time'], label=f'N_x = {Nx_value}', color=colors[Nx_value], marker='o')

plt.xlabel('N_t')
plt.ylabel('Computation Time (seconds)')
plt.title('Plot of Computation Time Across N_t for Different N_x Values')
plt.legend(title='N_x')
plt.yscale('log')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.savefig("computation_time_plot.png")
plt.close()

# Normalize the computation time by dividing by N_t * N_x^2
df['normalized_time'] = df['computation_time'] / (df['N_t'] * df['N_x'] ** 2)

# Save normalized computation time plot by N_x for different N_t values
plt.figure(figsize=(10, 6))
colors = {500: 'red', 1000: 'yellow', 3000: 'green', 5000: 'blue', 10000: 'purple', 15000: 'orange', 20000: 'brown'}
for Nt_value, group in df.groupby('N_t'):
    plt.plot(group['N_x'], group['normalized_time'], label=f'N_t = {Nt_value}', color=colors[Nt_value])

plt.xlabel('N_x')
plt.ylabel('Normalized Computation Time')
plt.title('Normalized Computation Time by N_x for Different N_t Values')
plt.legend(title="N_t Values")
plt.grid(True)
plt.savefig("normalized_computation_time_plot.png")
plt.close()
