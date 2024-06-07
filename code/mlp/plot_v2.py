import matplotlib.pyplot as plt
import pandas as pd

#fold_data = pd.DataFrame({
    #"Fold": range(10),
    #"Test Auc": [0.92496, 0.89788, 0.86310, 0.87853, 0.78824, 0.80924, 0.90095, 0.89420, 0.92364, 0.84600],
    #"Val Auc": [0.92779, 0.85531, 0.89980, 0.74405, 0.91006, 0.80820, 0.87969, 0.90923, 0.90153, 0.77076],
    #"Test Accuracy": [0.87179, 0.89474, 0.81579, 0.82051, 0.79130, 0.81416, 0.84483, 0.81739, 0.87069, 0.81538],
    #"Val Accuracy": [0.83333, 0.80992, 0.84677, 0.68254, 0.85345, 0.76423, 0.87179, 0.84483, 0.85470, 0.72358]
#})
#fold_data = pd.DataFrame({
    #"Fold": range(10),
    #"Test auc": [0.80291, 0.88978, 0.82069, 0.84289, 0.82322, 0.81005, 0.84006, 0.81753, 0.82813, 0.84650],
    #"Val auc": [0.80742, 0.86336, 0.81337, 0.83739, 0.81412, 0.82392, 0.81893, 0.81534, 0.80674, 0.85006],
    #"Test Accuracy": [0.62234, 0.68333, 0.63441, 0.67416, 0.64205, 0.60106, 0.66310, 0.63636, 0.62366, 0.69945],
    #"Val Accuracy": [0.58427, 0.71751, 0.61749, 0.65591, 0.64205, 0.65027, 0.61932, 0.64571, 0.63068, 0.65896]
#})
# Convert dictionary to DataFrame
dtype_spec = {
    'Test Accuracy': 'float64',
    'Test AUC': 'float64'
}
df = pd.read_csv('training_metrics.csv')
df['Test Accuracy'] = pd.to_numeric(df['Test Accuracy'], errors='coerce')
df['Test AUC'] = pd.to_numeric(df['Test AUC'], errors='coerce')
df = df.dropna()
print(df)
# Creating the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the accuracy lines with markers and annotations for each point
ax.plot(df['Fold'], df['Test Accuracy'], marker='o', linestyle='-', color='dodgerblue', label='Test Accuracy', linewidth=2, markersize=8)
ax.plot(df['Fold'], df['Test AUC'], marker='o', linestyle='-', color='coral', label='Test AUC', linewidth=2, markersize=8)

# Annotating each data point with its value
for i in df.index:
    ax.annotate(f"{df['Test Accuracy'][i]:.4f}", (df['Fold'][i], df['Test Accuracy'][i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(f"{df['Test AUC'][i]:.4f}", (df['Fold'][i], df['Test AUC'][i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Enhancing plot aesthetics
ax.set_title('Model Accuracy and AUC by Fold', fontsize=16, fontweight='bold', color='navy')
ax.set_xlabel('Fold Number', fontsize=14, fontweight='bold', color='darkgreen')
#ax.set_ylabel('Accuracy or AUC', fontsize=14, fontweight='bold', color='darkred')
ax.set_xticks(df['Fold'])  # Set x-ticks to match fold numbers
ax.set_ylim(0.0, 1.0)  # Set y-axis to focus on accuracy range
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('#f0f0f0')

plt.tight_layout()

# Save this plot as an SVG file
svg_file_path_corrected = "mlp_two_classes_metrics_annotated.svg"
plt.savefig(svg_file_path_corrected, format='svg', bbox_inches='tight')
plt.close()
