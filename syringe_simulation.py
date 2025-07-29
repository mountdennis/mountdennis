"""Syringe flow simulation with selectable geometry and viscosity."""

import numpy as np

# Parameters for viscosity models
mu0 = 1.0e-3
mu_inf = 1.0e-3
lam = 1.0
n_carreau = 0.6
k_power = 1.0e-3
n_power = 1.0


# Mapping from needle gauge to internal diameter (meters)
GAUGE_DIAMETERS = {
    "25G": 0.26e-3,
    "27G": 0.21e-3,
    "29G": 0.18e-3,
    "32G": 0.11e-3,
}


def viscosity_carreau(gamma):
    """Return viscosity from the Carreau model."""
    return mu_inf + (mu0 - mu_inf) * (1.0 + (lam * gamma) ** 2) ** ((n_carreau - 1.0) / 2.0)


def viscosity_power_law(gamma):
    """Return viscosity for a power-law fluid."""
    return k_power * gamma ** (n_power - 1.0)


def run_simulation(
    needle_length=12e-3,
    barrel_length=64e-3,
    barrel_diameter=6.35e-3,
    stopper_position=0.0,
    stopper_diameter=6.35e-3,
    gauge="27G",
    viscosity_model="carreau",
    taper_length=5e-3,
    taper_angle_deg=None,
):
    """Run the 2D syringe simulation.

    Parameters
    ----------
    needle_length : float
        Length of the needle in meters.
    barrel_length : float
        Total length of the barrel including any taper (meters).
    barrel_diameter : float
        Internal diameter of the barrel in meters.
    stopper_position : float
        Position of the stopper from the barrel base in meters.
    stopper_diameter : float
        Diameter of the stopper; must equal ``barrel_diameter``.
    gauge : str
        Needle gauge specifying the inner diameter.
    viscosity_model : str
        Either ``"carreau"`` or ``"power"``.
    taper_length : float, optional
        Length of the taper region in meters. Ignored if ``taper_angle_deg`` is
        provided.
    taper_angle_deg : float, optional
        If given, override ``taper_length`` by computing the taper length from
        this angle and the barrel/needle diameters.

    Returns
    -------
    float
        Average velocity magnitude inside the syringe at the end of the run.
    """

    if abs(barrel_diameter - stopper_diameter) > 1e-9:
        raise ValueError("Barrel and stopper diameters must be identical.")
    if gauge not in GAUGE_DIAMETERS:
        raise ValueError(f"Unknown gauge: {gauge}")
    diameter_needle = GAUGE_DIAMETERS[gauge]

    if taper_angle_deg is not None:
        taper_length = (
            barrel_diameter - diameter_needle
        ) / (2 * np.tan(np.radians(taper_angle_deg) / 2))

    length_cyl = barrel_length - stopper_position
    if length_cyl <= taper_length:
        raise ValueError("Stopper position leaves no straight section before taper.")

    length_total = length_cyl + needle_length

    nx, ny = 200, 80
    dx = length_total / (nx - 1)
    dy = barrel_diameter / (ny - 1)

    x = np.linspace(0, length_total, nx)

    def width(xi):
        if xi < length_cyl - taper_length:
            return barrel_diameter
        elif xi < length_cyl:
            r = (xi - (length_cyl - taper_length)) / taper_length
            return barrel_diameter + (diameter_needle - barrel_diameter) * r
        else:
            return diameter_needle

    width_x = np.array([width(xi) for xi in x])
    mask = np.zeros((ny, nx), dtype=bool)
    for j in range(nx):
        h = width_x[j]
        ny_active = int(h / dy)
        mask[:ny_active, j] = True

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    piston_velocity = 0.01

    def apply_bc():
        u[:, 0][mask[:, 0]] = piston_velocity
        v[:, 0] = 0
        u[~mask] = 0
        v[~mask] = 0
        for j in range(nx):
            ny_active = mask[:, j].sum()
            if ny_active > 0:
                u[ny_active - 1 :, j] = 0
                v[ny_active - 1 :, j] = 0
        u[:, -1] = u[:, -2]
        v[:, -1] = v[:, -2]
        p[:, -1] = p[:, -2]

    def shear_rate():
        du_dy = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dy
        dv_dx = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dx
        return np.sqrt(du_dy ** 2 + dv_dx ** 2) + 1e-12

    def pressure_poisson(b):
        for _ in range(50):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2
                + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2
                - b[1:-1, 1:-1] * dx**2 * dy**2
            ) / (2 * (dx**2 + dy**2))
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[-1, :] = 0
        return p

    nt = 100
    dt = 1e-4
    rho = 1000.0

    apply_bc()

    for _ in range(nt):
        un = u.copy()
        vn = v.copy()

        gamma = shear_rate()
        if viscosity_model == "carreau":
            mu = viscosity_carreau(gamma)
        else:
            mu = viscosity_power_law(gamma)
        nu_visc = mu / rho

        b = np.zeros_like(p)
        b[1:-1, 1:-1] = rho * (
            1 / dt * ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) + (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy))
            - ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx)) ** 2
            - 2 * ((un[2:, 1:-1] - un[:-2, 1:-1]) / (2 * dy) * (vn[1:-1, 2:] - vn[1:-1, :-2]) / (2 * dx))
            - ((vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy)) ** 2
        )

        p[:] = pressure_poisson(b)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu_visc
            * (
                dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
                + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu_visc
            * (
                dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
                + dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
            )
        )

        apply_bc()

    avg_vel = float(np.mean(u[mask]))
    print("Simulation complete. Final average velocity:", avg_vel)
    return avg_vel


if __name__ == "__main__":
    run_simulation()
