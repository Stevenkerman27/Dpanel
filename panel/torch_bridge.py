"""First-order CPU/float64 PyTorch bridge to the validated JAX panel interface.

This wraps, rather than reimplements, panel.differentiable.solve_fields.
It does not change Cp reconstruction or certify physical accuracy. Inputs must
belong to one already validated planar-wake/contact topology family.

Host copies are intentional. No GPU, DLPack, vmap, torch.compile, higher-order
AD or remeshing support is claimed. Backward recomputes the JAX VJP from saved
continuous inputs, instead of retaining a dense JAX pullback in PyTorch ctx.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import numpy as np
import torch
from torch.autograd.function import once_differentiable

from .differentiable import PanelTopology, solve_fields, validate_configuration


INPUT_NAMES = (
    "vertices", "station_y", "freestream", "density",
    "reference_area", "reference_chord", "reference_point",
)
OUTPUT_NAMES = (
    "Cp_raw", "panel_forces", "panel_moments", "CL", "CD_pressure", "CM",
)


class TorchPanelLoads(NamedTuple):
    """Raw pressure, SI panel loads and reference-dependent coefficients."""
    Cp_raw: torch.Tensor
    panel_forces: torch.Tensor
    panel_moments: torch.Tensor
    CL: torch.Tensor
    CD_pressure: torch.Tensor
    CM: torch.Tensor


def _host_copy(tensor: torch.Tensor) -> np.ndarray:
    # Copy: JAX and Torch must not alias mutable tensor storage across engines.
    return np.array(tensor.detach().numpy(), dtype=np.float64, copy=True, order="C")


class TorchPanelBridge:
    """One reusable compiled backend per frozen topology.

    Construct once, then pass new continuous Torch tensors on each call. Topology
    is snapshotted at construction and is not a trainable input. Geometry is
    checked at every forward, using the existing local configuration validator;
    this is NOT a complete CAD/wall-intersection audit.

    Inputs are unbatched torch.float64 CPU tensors. Scalars must have shape ().
    All seven inputs can carry first-order gradients. SI load units require SI
    geometry, velocity and density. No global JAX dtype/device flag is changed.
    """

    def __init__(self, topology: PanelTopology):
        if not jax.config.x64_enabled:
            raise ValueError("Enable jax_enable_x64 explicitly before constructing the bridge")
        if not isinstance(topology, PanelTopology):
            raise TypeError("Require a validated PanelTopology, not an inferred CAD mesh")
        self._cpu = jax.devices("cpu")[0]
        snapshot = []
        for value in topology:
            array = np.array(value, copy=True)
            array.setflags(write=False)
            snapshot.append(array)
        self._host_topology = PanelTopology(*snapshot)
        frozen = PanelTopology(*(jax.device_put(a, self._cpu) for a in snapshot))
        self._panel_count = len(snapshot[0])
        self._station_count = len(snapshot[5])
        if self._panel_count == 0:
            raise ValueError("The frozen topology has no wall panels")

        def selected(vertices, station_y, freestream, density, area, chord, point):
            out = solve_fields(vertices, station_y, frozen, freestream,
                               density, area, chord, point)
            return tuple(out[k] for k in OUTPUT_NAMES)

        def pullback(inputs, seeds):
            _, backward = jax.vjp(selected, *inputs)
            return backward(seeds)

        self._forward = jax.jit(selected)
        self._pullback = jax.jit(pullback)

    def __call__(self, vertices, station_y, freestream, density,
                 reference_area, reference_chord, reference_point):
        inputs = (vertices, station_y, freestream, density,
                  reference_area, reference_chord, reference_point)
        self._check_tensors(inputs)
        return TorchPanelLoads(*_TorchPanelFunction.apply(self, *inputs))

    def _check_tensors(self, inputs):
        if not jax.config.x64_enabled:
            raise ValueError("JAX x64 must remain enabled while using the bridge")
        for name, value in zip(INPUT_NAMES, inputs):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be an explicit torch.Tensor")
            if value.device.type != "cpu" or value.dtype != torch.float64:
                raise TypeError(f"{name} requires CPU torch.float64; no implicit cast or device transfer")
            if value.layout != torch.strided:
                raise TypeError(f"{name} must be a dense strided tensor")
        x, y, u, rho, area, chord, point = inputs
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("vertices must have shape [V,3]; batches are not supported")
        if y.shape != (self._station_count,) or u.shape != (3,) or point.shape != (3,):
            raise ValueError("Invalid station_y, freestream or reference_point shape")
        if any(v.shape != () for v in (rho, area, chord)):
            raise ValueError("density, reference_area and reference_chord must be scalar tensors")

    def _check_values(self, inputs):
        x, y, u, rho, area, chord, point = inputs
        if not all(np.isfinite(v).all() for v in inputs):
            raise ValueError("Nonfinite geometry, operating point or reference input")
        if rho <= 0 or area <= 0 or chord <= 0 or np.linalg.norm(u) == 0:
            raise ValueError("Require positive density/area/chord and nonzero freestream")
        if np.linalg.norm(np.cross(u, [0., 1., 0.])) == 0:
            raise ValueError("Freestream parallel to y gives an undefined lift axis")
        for name in ("faces", "wake_faces", "te_edges"):
            ix = np.asarray(getattr(self._host_topology, name))
            if ix.size == 0 or ix.min() < 0 or ix.max() >= len(x):
                raise ValueError(f"Frozen {name} indexes are incompatible with vertices")
        validate_configuration(x, y, self._host_topology)

    def _device_inputs(self, inputs):
        return tuple(jax.device_put(a, self._cpu) for a in inputs)


class _TorchPanelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, *inputs):
        arrays = tuple(_host_copy(x) for x in inputs)
        bridge._check_values(arrays)
        result = bridge._forward(*bridge._device_inputs(arrays))
        result = tuple(np.array(a, dtype=np.float64, copy=True) for a in result)
        if not all(np.isfinite(a).all() for a in result):
            raise FloatingPointError("Nonfinite JAX panel output; no clipping or silent fallback")
        ctx.bridge = bridge
        ctx.output_shapes = tuple(a.shape for a in result)
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)
        return tuple(torch.from_numpy(a) for a in result)

    @staticmethod
    @once_differentiable
    def backward(ctx, *output_grads):
        # Access via saved_tensors preserves PyTorch's in-place version checks.
        inputs = tuple(_host_copy(x) for x in ctx.saved_tensors)
        seeds = tuple(
            np.zeros(shape, dtype=np.float64) if g is None else _host_copy(g)
            for shape, g in zip(ctx.output_shapes, output_grads)
        )
        if not all(np.isfinite(g).all() for g in seeds):
            raise FloatingPointError("Nonfinite upstream load cotangent")
        bridge = ctx.bridge
        gradients = bridge._pullback(
            bridge._device_inputs(inputs), bridge._device_inputs(seeds)
        )
        values = tuple(np.array(g, dtype=np.float64, copy=True) for g in gradients)
        if not all(np.isfinite(g).all() for g in values):
            raise FloatingPointError("Nonfinite geometric VJP; no clipping or fallback")
        return (None,) + tuple(
            torch.from_numpy(g) if needed else None
            for g, needed in zip(values, ctx.needs_input_grad[1:])
        )
