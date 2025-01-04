import os
import matplotlib.pyplot as plt

plot_dir = "plot"  # Directory to save plots
os.makedirs(plot_dir, exist_ok=True)

def visualize_difference(differences, index):
    # Plot the differences
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(differences)), differences, label="Max Difference", color="blue", marker='o')
    plt.title("Difference Between Models")
    plt.xlabel("Test Index")
    plt.ylabel("Max Difference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/difference_between_models_{index}.png")  # Save plot with index
    plt.close()

# 画差异盒须图的函数
def plot_differences_box(differences, index):
    plt.figure(figsize=(10, 6))
    plt.boxplot(differences, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'), capprops=dict(color='blue'), flierprops=dict(markerfacecolor='red', marker='o', markersize=6))
    plt.title("Box Plot of Differences Between Models")
    plt.xlabel("Max Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/box_plot_of_differences_{index}.png")  # Save plot with index
    plt.close()