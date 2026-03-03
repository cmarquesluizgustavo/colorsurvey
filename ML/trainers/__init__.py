from .base import BaseTrainer, TrainerFactory
from .metric_learning_trainer import MetricLearningTrainer
from .xgboost_trainer import XGBoostTrainer
from .color_clip_trainer import ColorCLIPTrainer

__all__ = [
    "BaseTrainer",
    "TrainerFactory",
    "MetricLearningTrainer",
    "XGBoostTrainer",
    "ColorCLIPTrainer",
]
