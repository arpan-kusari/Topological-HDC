from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
 
def plot_classification_heatmap(confusion_matrix, class_labels=range(0,10), figsize=(8, 6), title="Confusion Matrix", fig_filename='figures/confusion_matrix.png', fontsize=20, dpi=150) -> None:
    """
    Plots a heatmap for a classification confusion matrix.

    Parameters:
    - confusion_matrix: 2D NumPy array (square) representing classification results.
    - class_labels: List of labels for classes (default: None).
    - title: Title of the heatmap plot (default: "Classification Heatmap").
    """
    plt.figure(figsize=figsize)  # Set figure size
    sns.heatmap(confusion_matrix, annot=True, fmt="", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, annot_kws={"size": fontsize})

    plt.xlabel("Predicted Labels", fontsize=fontsize)
    plt.ylabel("True Labels", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    folder = Path(fig_filename).parent
    folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_filename, dpi=dpi) 
