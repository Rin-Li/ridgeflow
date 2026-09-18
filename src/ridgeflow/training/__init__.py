"""Rectified-flow training for the ridge heatmap model."""

from ridgeflow.training.dataset import RidgeDataset
from ridgeflow.training.trainer import TrainConfig, train

__all__ = ["RidgeDataset", "TrainConfig", "train"]
