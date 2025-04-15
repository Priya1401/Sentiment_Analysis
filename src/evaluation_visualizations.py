#!/usr/bin/env python
"""
evaluation_visualizations.py

This module provides functions to generate evaluation visualizations for sentiment
analysis models. It includes functions to print a classification report and
to plot & save a confusion matrix as a PNG file.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, labels, model_name, save_path=None):
    """
    Generate and save a confusion matrix heatmap.

    Parameters:
    - y_true: Array-like of true labels.
    - y_pred: Array-like of predicted labels.
    - labels: List of label names (strings).
    - model_name: String used for the plot title and for naming the saved file.
    - save_path: Optional file path to save the plot. If None, a default name is used.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")

    if save_path is None:
        save_path = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def print_classification_report(y_true, y_pred, labels, model_name):
    """
    Print and return the classification report for the specified model.

    Parameters:
    - y_true: Array-like of true labels.
    - y_pred: Array-like of predicted labels.
    - labels: List of label names.
    - model_name: String representing the model name.

    Returns:
    - report: The text classification report.
    """
    report = classification_report(y_true, y_pred, target_names=labels)
    print(f"Classification Report for {model_name}:\n{report}")
    return report


def evaluate_model(y_true, y_pred, labels, model_name, save_cm=True):
    """
    Generate a classification report and, optionally, a confusion matrix visualization.

    Parameters:
    - y_true: Array-like, true sentiment labels.
    - y_pred: Array-like, predicted sentiment labels.
    - labels: List of sentiment labels as strings.
    - model_name: Name of the model (e.g., "Logistic Regression").
    - save_cm: Boolean indicating whether to save the confusion matrix plot (default is True).

    Returns:
    - report: The classification report string.
    """
    report = print_classification_report(y_true, y_pred, labels, model_name)
    if save_cm:
        plot_confusion_matrix(y_true, y_pred, labels, model_name)
    return report


# Example usage (for testing this module directly)
if __name__ == "__main__":
    # Sample data (this example assumes three classes: Negative, Neutral, Positive)
    y_true = [0, 1, 2, 2, 1, 0, 1, 2]
    y_pred = [0, 2, 2, 2, 1, 0, 0, 2]
    labels = ["Negative", "Neutral", "Positive"]

    evaluate_model(y_true, y_pred, labels, "Test Model")
