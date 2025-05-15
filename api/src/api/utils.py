"""Utility functions for non-AI analysis"""

import base64
import typing as tp

import numpy as np


def float_nanmean(num_list: np.ndarray | tp.List[float]) -> float:
    """Mean the numbers of a list, ignoring nan and returning as a float"""
    return float(np.nanmean(num_list))


def from_base64(string: str):
    """Convert a string response from the frontend to data"""
    data = string.split(",")[1]
    return base64.b64decode(data)


def clip(array: np.ndarray):
    """Clip the confidence arrays to binary masks."""
    return (array > 0.5).astype(np.uint8)
