import matplotlib.pyplot as plt
import pandas as pd
# Assuming the given data is correctly formatted as repeated rows are a misinterpretation, let's process it correctly
# We should combine the repeated entries for fold 0 by averaging them for demonstration purposes
fold_data = {
    "Fold": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Val Accuracy": [
        0.6952381134033203,  # Average for fold 0
        0.6428571939468384, 0.7095238566398621, 0.6666666865348816,
        0.6857143044471741, 0.6809524297714233, 0.6428571939468384,
        0.6285714507102966, 0.6761904954910278, 0.6142857670783997
    ],
    "Test Accuracy": [
        0.7028301954269409,  # Average for fold 0
        0.698113203048706, 0.650943398475647, 0.6462264060974121,
        0.6255924701690674, 0.6729857921600342, 0.6966825127601624,
        0.6208531260490417, 0.6161137819290161, 0.6255924701690674
    ]
}
# Convert dictionary to DataFrame
df = pd.DataFrame(fold_data)

# Creating the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the accuracy lines with markers and annotations for each point
ax.plot(df['Fold'], df['Val Accuracy'], marker='o', linestyle='-', color='dodgerblue', label='Validation Accuracy', linewidth=2, markersize=8)
ax.plot(df['Fold'], df['Test Accuracy'], marker='o', linestyle='-', color='coral', label='Test Accuracy', linewidth=2, markersize=8)

# Annotating each data point with its value
for i in df.index:
    ax.annotate(f"{df['Val Accuracy'][i]:.4f}", (df['Fold'][i], df['Val Accuracy'][i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax.annotate(f"{df['Test Accuracy'][i]:.4f}", (df['Fold'][i], df['Test Accuracy'][i]), textcoords="offset points", xytext=(0,-15), ha='center')

# Enhancing plot aesthetics
ax.set_title('Model Accuracy by Fold', fontsize=16, fontweight='bold', color='navy')
ax.set_xlabel('Fold Number', fontsize=14, fontweight='bold', color='darkgreen')
ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='darkred')
ax.set_xticks(df['Fold'])  # Set x-ticks to match fold numbers
ax.set_ylim(0.6, 0.75)  # Set y-axis to focus on accuracy range
ax.legend()
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_facecolor('#f0f0f0')

plt.tight_layout()

# Save this plot as an SVG file
svg_file_path_corrected = "metrics_annotated_v2.svg"
plt.savefig(svg_file_path_corrected, format='svg', bbox_inches='tight')
plt.close()

