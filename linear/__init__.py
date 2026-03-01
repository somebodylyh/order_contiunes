"""
Linear Rotation-Accumulation Experiment

A toy task for testing Transformer's long-range dependency and order discovery
capabilities using orthogonal matrix dynamics.

Phase 0: Data quality validation (current)
Phase 1: Full training pipeline (conditional on Phase 0 success)
"""

from .data_generator import LinearDynamicalGenerator

__all__ = ['LinearDynamicalGenerator']
