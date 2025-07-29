"""Tkinter GUI for configuring and running the syringe simulation."""

import tkinter as tk
from tkinter import ttk, messagebox

from syringe_simulation import run_simulation, GAUGE_DIAMETERS


def main():
    root = tk.Tk()
    root.title("Syringe Simulation")

    tk.Label(root, text="Needle length (mm)").grid(row=0, column=0, sticky="w")
    length_var = tk.StringVar(value="12")
    tk.Entry(root, textvariable=length_var, width=10).grid(row=0, column=1)

    tk.Label(root, text="Barrel diameter (mm)").grid(row=1, column=0, sticky="w")
    barrel_var = tk.StringVar(value="6.35")
    tk.Entry(root, textvariable=barrel_var, width=10).grid(row=1, column=1)

    tk.Label(root, text="Stopper diameter (mm)").grid(row=2, column=0, sticky="w")
    stopper_var = tk.StringVar(value="6.35")
    tk.Entry(root, textvariable=stopper_var, width=10).grid(row=2, column=1)

    tk.Label(root, text="Needle gauge").grid(row=3, column=0, sticky="w")
    gauge_var = tk.StringVar(value="27G")
    ttk.OptionMenu(root, gauge_var, "27G", *GAUGE_DIAMETERS.keys()).grid(row=3, column=1, sticky="ew")

    tk.Label(root, text="Viscosity model").grid(row=4, column=0, sticky="w")
    model_var = tk.StringVar(value="carreau")
    ttk.OptionMenu(root, model_var, "carreau", "carreau", "power").grid(row=4, column=1, sticky="ew")

    result_var = tk.StringVar()
    tk.Label(root, textvariable=result_var).grid(row=6, column=0, columnspan=2, pady=5)

    def run():
        try:
            needle_length = float(length_var.get()) * 1e-3
            barrel_d = float(barrel_var.get()) * 1e-3
            stopper_d = float(stopper_var.get()) * 1e-3
            avg = run_simulation(
                needle_length=needle_length,
                barrel_diameter=barrel_d,
                stopper_diameter=stopper_d,
                gauge=gauge_var.get(),
                viscosity_model=model_var.get(),
            )
            result_var.set(f"Average velocity: {avg:.6f} m/s")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    ttk.Button(root, text="Run Simulation", command=run).grid(row=5, column=0, columnspan=2, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
