"""
models package — Multi-Task Model Architecture
================================================
Contains the backbone, task-specific heads, and the combined multi-task model.
"""

from .backbone import CustomCNNBackbone, PretrainedBackbone
from .classifier_head import ClassifierHead
from .decoder_head import DecoderHead
from .multitask_model import MultiTaskModel

__all__ = [
    "CustomCNNBackbone",
    "PretrainedBackbone",
    "ClassifierHead",
    "DecoderHead",
    "MultiTaskModel",
]
