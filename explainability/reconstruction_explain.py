import torch
import numpy as np
import matplotlib.pyplot as plt


FEATURE_NAMES = ["meantemp", "humidity", "wind_speed", "meanpressure"]


def reconstruction_error_map(model, x):
    """
    Computes reconstruction error per time-step and per feature.

    Args:
        model: trained autoencoder
        x: input tensor (1, T, F)

    Returns:
        error_map: (T, F) numpy array
    """
    model.eval()
    with torch.no_grad():
        x_hat = model(x)
        error = (x - x_hat) ** 2

    return error.squeeze(0).cpu().numpy()


def explain_window(model, x):
    """
    Generates explainability metrics for a single window.
    """
    error_map = reconstruction_error_map(model, x)

    feature_importance = error_map.sum(axis=0)
    temporal_importance = error_map.sum(axis=1)

    return error_map, feature_importance, temporal_importance


def plot_explanation(error_map, feature_importance, temporal_importance):
    """
    Visualization for explainability.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # Heatmap (Time × Feature)
    im = axs[0].imshow(error_map.T, aspect="auto", cmap="hot")
    axs[0].set_title("Reconstruction Error Heatmap")
    axs[0].set_ylabel("Features")
    axs[0].set_yticks(range(len(FEATURE_NAMES)))
    axs[0].set_yticklabels(FEATURE_NAMES)
    plt.colorbar(im, ax=axs[0])

    # Feature importance
    axs[1].bar(FEATURE_NAMES, feature_importance)
    axs[1].set_title("Feature-wise Contribution")

    # Temporal importance
    axs[2].plot(temporal_importance)
    axs[2].set_title("Temporal Contribution")
    axs[2].set_xlabel("Time Step")

    plt.tight_layout()
    plt.show()
