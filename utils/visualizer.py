import matplotlib.pyplot as plt

def visualize_difference(differences):
    """可视化差异"""
    plt.plot(differences)
    plt.title("Difference Between Models")
    plt.xlabel("Test Index")
    plt.ylabel("Max Difference")
    plt.show()
