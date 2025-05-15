import matplotlib.pyplot as plt
import numpy as np

# ELPD data
elpd_per_degree = {"1": -1.277, "2": -1.082, "3": -0.916, "4": -0.949}

# Extract x and y values
degrees = [int(k) for k in elpd_per_degree.keys()]
elpd_values = [elpd_per_degree[k] for k in elpd_per_degree.keys()]

# Create the plot with similar styling to previous visualization
plt.style.use("seaborn-v0_8-whitegrid")

# Define colors and sizes
main_color = "#1f77b4"  # Blue
highlight_color = "#ff7f0e"  # Orange
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12

# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot the line
ax.plot(
    degrees,
    elpd_values,
    marker="o",
    markersize=10,
    linewidth=2,
    color=main_color,
    alpha=0.8,
)

# Highlight the best value (maximum ELPD)
best_idx = np.argmax(elpd_values)
best_degree = degrees[best_idx]
best_value = elpd_values[best_idx]
ax.scatter(
    [best_degree],
    [best_value],
    s=150,
    color=highlight_color,
    edgecolor="white",
    linewidth=2,
    zorder=10,
    label=f"Optimal degree: {best_degree}",
)

# Annotate the best value
ax.annotate(
    f"Avg_LPD: {best_value:.3f}",
    xy=(best_degree, best_value),
    xytext=(best_degree, best_value + 0.025),
    fontsize=12,
    # arrowprops=dict(facecolor="black", shrink=0.01, width=1.5, headwidth=8, alpha=0.7),
)

# Add data point values
for i, (degree, value) in enumerate(zip(degrees, elpd_values)):
    if i != best_idx:  # Skip the best value as it has its own annotation
        ax.annotate(
            f"{value:.3f}",
            xy=(degree, value),
            xytext=(0, -15),  # 15 points vertical offset downward
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

# Customize axes
ax.set_xticks(degrees)
ax.set_xticklabels([str(d) for d in degrees])
ax.set_xlabel("Polynomial Degree", fontsize=LABEL_SIZE)
ax.set_ylabel("Average LPD Value", fontsize=LABEL_SIZE)
ax.set_title(
    "Average Log Predictive Density by Polynomial Degree",
    fontsize=TITLE_SIZE,
    fontweight="bold",
)
ax.tick_params(axis="both", labelsize=TICK_SIZE)

# Add a gray horizontal line to show the maximum
ax.axhline(y=best_value, color="gray", linestyle="--", alpha=0.5)

# Add legend
ax.legend(fontsize=LEGEND_SIZE, loc="lower right")

# Set y limits to focus on data range with some padding
y_min = min(elpd_values) - 0.05
y_max = max(elpd_values) + 0.05
ax.set_ylim(y_min, y_max)

# Ensure a professional appearance
plt.tight_layout()

# Save the figure
plt.savefig(
    "PFAS_HalfLife_QSAR/ED_GPR/Revamped_dataset/elpd_per_degree_original.png",
    dpi=300,
    bbox_inches="tight",
)

# Show the plot
plt.show()
