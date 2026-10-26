"""Experimental PyTorch boundary-element solvers and geometry-differentiable kernels.

Independent, validated nonlifting primitives: torch_panel.kernels,
torch_panel.symmetry, torch_panel.nonlifting.

Opt-in CPU/CUDA Float64 lifting RESEARCH backend:
torch_panel.half_lifting (right-half source/doublet, finite wake, weak
Kutta, raw Cp, dense LU and explicit first-order implicit adjoint).

This package is not wired into the production Dpanel solve.py or GRASS
training, and successful Torch/JAX numerical parity does not establish
aerodynamic physical accuracy.
"""
