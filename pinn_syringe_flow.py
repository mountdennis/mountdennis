"""Physics-Informed Neural Network for laminar syringe-needle flow.

This script implements a parametric PINN capable of handling multiple
syringe/needle geometries, viscosity models, and driving conditions.
The geometry parameters are symbolic placeholders so actual values can
be substituted later without changing the code structure.

Run with ``--ask-geometry`` to interactively answer questions about
barrel diameter, needle gauge, and needle length before training.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torch.autograd import grad


torch.manual_seed(0)


@dataclass
class GeometryConfig:
    """Container for syringe and needle geometry settings.

    The length and radius values should be provided externally. Symbols
    are defined for supported configurations but can be updated without
    modifying the rest of the code.
    """

    L_syr: float
    R_syr: float
    L_needle: float
    R_needle: float


@dataclass
class FluidConfig:
    """Fluid property configuration supporting Newtonian and power-law models."""

    viscosity: float = 1.0  # Effective viscosity for Newtonian fluid
    power_law_K: float = 1.0  # Consistency index (for power-law)
    power_law_n: float = 1.0  # Flow behavior index; n < 1 implies shear-thinning
    use_power_law: bool = False


@dataclass
class DrivingConfig:
    """Driving boundary condition configuration."""

    mode: str = "velocity"  # "velocity" or "pressure"
    plunger_velocity: float = 1.0  # Representative velocity scale
    plunger_pressure: float = 1.0  # Representative pressure scale
    outlet_pressure: float = 0.0

    def inlet_velocity(self, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent plunger velocity profile.

        The default implementation is constant, but any time function
        returning a tensor with the same shape as ``t`` can be supplied.
        """

        return self.plunger_velocity * torch.ones_like(t)

    def inlet_pressure(self, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent plunger pressure profile."""

        return self.plunger_pressure * torch.ones_like(t)


@dataclass
class SyringeNeedleConfig:
    geometry: GeometryConfig
    fluid: FluidConfig
    driving: DrivingConfig


# Symbolic geometry placeholders (to be set with real dimensions externally)
L_syr_2p25 = 1.0
R_syr_2p25 = 1.0
L_syr_1p0 = 1.0
R_syr_1p0 = 1.0
L_needle_symbolic = 1.0
R_needle_27G = 1.0
R_needle_29G = 1.0


CONFIGURATIONS: Dict[str, SyringeNeedleConfig] = {
    "2.25mL_27G": SyringeNeedleConfig(
        geometry=GeometryConfig(
            L_syr=L_syr_2p25,
            R_syr=R_syr_2p25,
            L_needle=L_needle_symbolic,
            R_needle=R_needle_27G,
        ),
        fluid=FluidConfig(),
        driving=DrivingConfig(mode="velocity"),
    ),
    "2.25mL_29G": SyringeNeedleConfig(
        geometry=GeometryConfig(
            L_syr=L_syr_2p25,
            R_syr=R_syr_2p25,
            L_needle=L_needle_symbolic,
            R_needle=R_needle_29G,
        ),
        fluid=FluidConfig(),
        driving=DrivingConfig(mode="velocity"),
    ),
    "1.0mL_29G": SyringeNeedleConfig(
        geometry=GeometryConfig(
            L_syr=L_syr_1p0,
            R_syr=R_syr_1p0,
            L_needle=L_needle_symbolic,
            R_needle=R_needle_29G,
        ),
        fluid=FluidConfig(),
        driving=DrivingConfig(mode="velocity"),
    ),
}


# Catalog of common hypodermic needle gauges (inner diameter, mm) for quick selection
GAUGE_INNER_DIAMETERS_MM: Dict[str, float] = {
    "25": 0.26,
    "26": 0.241,
    "27": 0.21,
    "28": 0.184,
    "29": 0.159,
    "30": 0.133,
    "31": 0.114,
    "32": 0.108,
}


def _prompt_float(prompt: str, default: float) -> float:
    """Prompt the user for a float, falling back to default on empty input."""

    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, using default.")
        return default


def _prompt_gauge(default: str) -> str:
    """Prompt the user for a needle gauge using the catalog table."""

    print("\nSelect needle gauge (inner diameters in mm):")
    for g, d in sorted(GAUGE_INNER_DIAMETERS_MM.items(), key=lambda x: int(x[0])):
        print(f"  {g}G -> {d} mm")
    raw = input(f"Needle gauge [default {default}]: ").strip()
    gauge = raw if raw else default
    if gauge not in GAUGE_INNER_DIAMETERS_MM:
        print("Gauge not in catalog; retaining default.")
        gauge = default
    return gauge


def prompt_geometry_updates(base: SyringeNeedleConfig) -> SyringeNeedleConfig:
    """Ask the user for barrel diameter, needle gauge, and needle length.

    The dialog is optional and will be skipped when stdin is not a TTY
    (e.g., in automated tests). Returned configuration includes updated
    radii/lengths while preserving other settings.
    """

    if not sys.stdin.isatty():
        return base

    geom = base.geometry
    print("\nProvide syringe/needle geometry. Press Enter to keep defaults.")
    barrel_diam_mm = _prompt_float("Barrel inner diameter (mm)", default=2 * geom.R_syr)
    L_syr_mm = _prompt_float("Syringe length (mm)", default=geom.L_syr)
    needle_len_mm = _prompt_float("Needle length (mm)", default=geom.L_needle)
    gauge = _prompt_gauge("29")
    needle_id_mm = GAUGE_INNER_DIAMETERS_MM[gauge]

    updated_geometry = GeometryConfig(
        L_syr=L_syr_mm,
        R_syr=barrel_diam_mm / 2.0,
        L_needle=needle_len_mm,
        R_needle=needle_id_mm / 2.0,
    )
    return SyringeNeedleConfig(geometry=updated_geometry, fluid=base.fluid, driving=base.driving)


def radius_profile(z: torch.Tensor, geometry: GeometryConfig) -> torch.Tensor:
    """Return the local radius as a function of axial position.

    Syringe: 0 <= z <= L_syr. Needle: L_syr < z <= L_syr + L_needle. The
    function allows the PINN to evaluate boundary conditions without
    hard-coding specific radii in different regions.
    """

    return torch.where(z <= geometry.L_syr, geometry.R_syr, geometry.R_needle)


class FullyConnectedNN(nn.Module):
    """Simple feed-forward network with configurable hidden layers."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 3))  # u_z, u_r, p
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(inputs)
        u_z = out[:, 0:1]
        u_r = out[:, 1:2]
        p = out[:, 2:3]
        return u_z, u_r, p


def build_pinn(in_dim: int) -> FullyConnectedNN:
    """Build the PINN model."""

    return FullyConnectedNN(in_dim)


def viscosity_from_shear_rate(
    duz_dr: torch.Tensor,
    dur_dz: torch.Tensor,
    fluid: FluidConfig,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute viscosity using either Newtonian or power-law model."""

    if not fluid.use_power_law:
        return fluid.viscosity * torch.ones_like(duz_dr)
    gamma_dot = torch.sqrt(2.0 * duz_dr**2 + 2.0 * dur_dz**2) + eps
    return fluid.power_law_K * gamma_dot ** (fluid.power_law_n - 1.0)


def pde_residuals(
    model: FullyConnectedNN,
    inputs: torch.Tensor,
    config: SyringeNeedleConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Navier–Stokes and continuity residuals using autograd.

    Inputs tensor should contain normalized z, r, t as well as the
    parameter vector. Gradients are taken with respect to the first
    three columns (z, r, t) only.
    """

    inputs.requires_grad_(True)
    u_z, u_r, p = model(inputs)

    z = inputs[:, 0:1]
    r = inputs[:, 1:2]
    t = inputs[:, 2:3]

    def grads(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        value = grad(y, x, torch.ones_like(y), create_graph=True, allow_unused=True)[0]
        if value is None:
            value = torch.zeros_like(x, requires_grad=True)
        return value

    duz_dz = grads(u_z, z)
    duz_dr = grads(u_z, r)
    duz_dt = grads(u_z, t)

    dur_dz = grads(u_r, z)
    dur_dr = grads(u_r, r)
    dur_dt = grads(u_r, t)

    dp_dz = grads(p, z)
    dp_dr = grads(p, r)

    mu = viscosity_from_shear_rate(duz_dr, dur_dz, config.fluid)

    # Axisymmetric continuity: 1/r * ∂(r u_r)/∂r + ∂u_z/∂z = 0
    continuity = (1.0 / (r + 1e-6)) * (u_r + r * dur_dr) + duz_dz

    # Viscous terms in cylindrical coordinates (neglecting swirl)
    laplace_uz = grads(duz_dz, z) + (1.0 / (r + 1e-6)) * duz_dr + grads(duz_dr, r)
    laplace_ur = (
        grads(dur_dz, z)
        + (1.0 / (r + 1e-6)) * dur_dr
        + grads(dur_dr, r)
        - u_r / ((r + 1e-6) ** 2)
    )

    inertia_z = duz_dt + u_z * duz_dz + u_r * duz_dr
    inertia_r = dur_dt + u_z * dur_dz + u_r * dur_dr - u_r**2 / (r + 1e-6)

    momentum_z = inertia_z + dp_dz / config.fluid.viscosity - laplace_uz
    momentum_r = inertia_r + dp_dr / config.fluid.viscosity - laplace_ur

    return continuity, momentum_z, momentum_r


def wall_bc(model: FullyConnectedNN, z: torch.Tensor, geometry: GeometryConfig, params: torch.Tensor) -> torch.Tensor:
    """No-slip wall boundary condition at r = radius(z)."""

    r_wall = radius_profile(z, geometry)
    inputs = torch.cat([z, r_wall, torch.zeros_like(z), params], dim=1)
    u_z, u_r, _ = model(inputs)
    return torch.mean(u_z**2 + u_r**2)


def axis_bc(model: FullyConnectedNN, r: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Axis symmetry at r = 0."""

    z = torch.rand_like(r)
    inputs = torch.cat([z, torch.zeros_like(r), torch.zeros_like(r), params], dim=1)
    inputs.requires_grad_(True)
    u_z, u_r, _ = model(inputs)
    grads = lambda y, x: grad(
        y, x, torch.ones_like(y), create_graph=True, allow_unused=True
    )[0]
    duz_dr = grads(u_z, inputs[:, 1:2])
    if duz_dr is None:
        duz_dr = torch.zeros_like(inputs[:, 1:2])
    return torch.mean(u_r**2 + duz_dr**2)


def inlet_bc(
    model: FullyConnectedNN,
    geometry: GeometryConfig,
    driving: DrivingConfig,
    params: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Inlet boundary at plunger side (z = 0). Supports velocity or pressure modes."""

    r = torch.rand(num_points, 1) * geometry.R_syr
    t = torch.rand(num_points, 1)
    z = torch.zeros_like(r)
    inputs = torch.cat([z, r, t, params.expand(num_points, -1)], dim=1)
    u_z, u_r, p = model(inputs)

    if driving.mode == "velocity":
        target = driving.inlet_velocity(t)
        loss = torch.mean((u_z - target) ** 2) + torch.mean(u_r**2)
    else:
        target_p = driving.inlet_pressure(t)
        loss = torch.mean((p - target_p) ** 2) + torch.mean(u_r**2)
    return loss


def outlet_bc(
    model: FullyConnectedNN,
    geometry: GeometryConfig,
    driving: DrivingConfig,
    params: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Pressure outlet at needle tip."""

    r = torch.rand(num_points, 1) * geometry.R_needle
    t = torch.rand(num_points, 1)
    z = torch.full_like(r, geometry.L_syr + geometry.L_needle)
    inputs = torch.cat([z, r, t, params.expand(num_points, -1)], dim=1)
    _, _, p = model(inputs)
    target_p = driving.outlet_pressure * torch.ones_like(p)
    return torch.mean((p - target_p) ** 2)


def sample_collocation(
    geometry: GeometryConfig, num_points: int, needle_bias: float = 0.3
) -> torch.Tensor:
    """Sample interior collocation points with bias towards the needle region."""

    num_needle = int(num_points * needle_bias)
    num_syringe = num_points - num_needle

    z_syr = torch.rand(num_syringe, 1) * geometry.L_syr
    r_syr = torch.rand(num_syringe, 1) * geometry.R_syr

    z_needle = geometry.L_syr + torch.rand(num_needle, 1) * geometry.L_needle
    r_needle = torch.rand(num_needle, 1) * geometry.R_needle

    z = torch.cat([z_syr, z_needle], dim=0)
    r = torch.cat([r_syr, r_needle], dim=0)
    t = torch.rand(num_points, 1)
    return torch.cat([z, r, t], dim=1)


def normalize_inputs(inputs: torch.Tensor, geometry: GeometryConfig) -> torch.Tensor:
    """Normalize z and r to [0, 1] using geometry scales."""

    z = inputs[:, 0:1] / (geometry.L_syr + geometry.L_needle)
    r = inputs[:, 1:2] / geometry.R_syr
    t = inputs[:, 2:3]
    return torch.cat([z, r, t], dim=1)


def make_parameter_vector(config: SyringeNeedleConfig) -> torch.Tensor:
    """Assemble parameter vector for the PINN input."""

    geometry = config.geometry
    fluid = config.fluid
    driving = config.driving
    params = torch.tensor(
        [
            geometry.L_syr,
            geometry.R_syr,
            geometry.L_needle,
            geometry.R_needle,
            fluid.viscosity,
            fluid.power_law_K,
            fluid.power_law_n,
            float(fluid.use_power_law),
            driving.plunger_velocity,
            driving.plunger_pressure,
            driving.outlet_pressure,
            1.0 if driving.mode == "pressure" else 0.0,
        ]
    ).float()
    return params


def loss_function(
    model: FullyConnectedNN,
    config: SyringeNeedleConfig,
    collocation_points: torch.Tensor,
    params: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute combined loss terms for PDE and boundary conditions."""

    geometry = config.geometry
    driving = config.driving

    normalized = normalize_inputs(collocation_points, geometry)
    param_vec = params.expand(collocation_points.shape[0], -1)
    inputs = torch.cat([normalized, param_vec], dim=1)
    inputs.requires_grad_(True)

    cont_res, mom_z_res, mom_r_res = pde_residuals(model, inputs, config)
    pde_loss = torch.mean(cont_res**2) + torch.mean(mom_z_res**2) + torch.mean(mom_r_res**2)

    # Boundary condition losses
    z_wall = torch.rand(64, 1) * (geometry.L_syr + geometry.L_needle)
    wall_loss = wall_bc(model, z_wall, geometry, param_vec[: z_wall.shape[0]])
    axis_loss = axis_bc(model, torch.rand(64, 1), param_vec[:64])
    inlet_loss = inlet_bc(model, geometry, driving, param_vec[0:1], num_points=64)
    outlet_loss = outlet_bc(model, geometry, driving, param_vec[0:1], num_points=64)

    total_loss = pde_loss + wall_loss + axis_loss + inlet_loss + outlet_loss
    components = {
        "pde": pde_loss.detach(),
        "wall": wall_loss.detach(),
        "axis": axis_loss.detach(),
        "inlet": inlet_loss.detach(),
        "outlet": outlet_loss.detach(),
    }
    return total_loss, components


def train(model: FullyConnectedNN, config: SyringeNeedleConfig, num_iters: int = 1000) -> None:
    """Simple training loop for the PINN."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    params = make_parameter_vector(config)
    geometry = config.geometry

    for itr in range(num_iters):
        collocation = sample_collocation(geometry, num_points=256)
        loss, comps = loss_function(model, config, collocation, params)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            msg = (
                f"Iter {itr}: loss={loss.item():.4e}, "
                f"pde={comps['pde'].item():.4e}, wall={comps['wall'].item():.4e}, "
                f"axis={comps['axis'].item():.4e}, inlet={comps['inlet'].item():.4e}, "
                f"outlet={comps['outlet'].item():.4e}"
            )
            print(msg)


def evaluate_outlet_profile(
    model: FullyConnectedNN, config: SyringeNeedleConfig, num_points: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate velocity and pressure at the needle outlet."""

    geometry = config.geometry
    params = make_parameter_vector(config)
    z = torch.full((num_points, 1), geometry.L_syr + geometry.L_needle)
    r = torch.linspace(0, geometry.R_needle, num_points).unsqueeze(1)
    t = torch.zeros_like(z)
    inputs = torch.cat([z, r, t], dim=1)
    normalized = normalize_inputs(inputs, geometry)
    inp = torch.cat([normalized, params.expand(num_points, -1)], dim=1)
    u_z, _, p = model(inp)
    return u_z.detach(), p.detach()


def compute_flowrate(u_z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Compute volumetric flowrate by integrating axial velocity over area."""

    # Simple trapezoidal integration in radial direction for axisymmetric flow
    dr = r[1] - r[0]
    integrand = 2 * math.pi * r * u_z
    return torch.trapz(integrand.squeeze(), dx=dr.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN for syringe-needle flow")
    parser.add_argument("--ask-geometry", action="store_true", help="prompt for barrel diameter, gauge, and needle length")
    parser.add_argument("--num-iters", type=int, default=200, help="training iterations for demonstration")
    args = parser.parse_args()

    # Select configuration and update parameters as needed
    base_config = CONFIGURATIONS["1.0mL_29G"]
    config = SyringeNeedleConfig(
        geometry=base_config.geometry,
        fluid=FluidConfig(viscosity=1.0, power_law_K=1.0, power_law_n=0.6, use_power_law=True),
        driving=DrivingConfig(mode="velocity", plunger_velocity=1.0, outlet_pressure=0.0),
    )

    if args.ask_geometry:
        config = prompt_geometry_updates(config)

    input_dim = 3 + 12  # z, r, t + parameter vector length
    model = build_pinn(input_dim)

    print("Starting training (few iterations for demonstration)...")
    train(model, config, num_iters=args.num_iters)

    print("Evaluating outlet profile...")
    u_z, p = evaluate_outlet_profile(model, config)
    geometry = config.geometry
    r = torch.linspace(0, geometry.R_needle, u_z.shape[0]).unsqueeze(1)
    flowrate = compute_flowrate(u_z, r)
    delta_p = p.max() - p.min()

    print(f"Max outlet axial velocity: {u_z.max().item():.4f}")
    print(f"Outlet pressure range: [{p.min().item():.4f}, {p.max().item():.4f}], drop={delta_p.item():.4f}")
    print(f"Estimated volumetric flowrate: {flowrate.item():.4f}")

    # Demonstrate switching configuration without altering network structure
    alt_config = CONFIGURATIONS["2.25mL_27G"]
    alt_params = make_parameter_vector(alt_config)
    sample_points = torch.tensor([[0.5 * alt_config.geometry.L_syr, 0.1 * alt_config.geometry.R_syr, 0.0]])
    normalized = normalize_inputs(sample_points, alt_config.geometry)
    inputs = torch.cat([normalized, alt_params.unsqueeze(0)], dim=1)
    u_z_alt, u_r_alt, p_alt = model(inputs)
    print("Example evaluation for alternate geometry (untrained weights reused):")
    print(
        f"u_z={u_z_alt.item():.4f}, u_r={u_r_alt.item():.4f}, "
        f"p={p_alt.item():.4f} at mid-syringe location"
    )


if __name__ == "__main__":
    main()
