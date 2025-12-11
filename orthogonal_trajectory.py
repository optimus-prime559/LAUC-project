"""
orthogonal_trajectory.py

A self-contained Python script to compute and plot orthogonal trajectories
for a given one-parameter family of curves.

Features
- Symbolically computes orthogonal trajectories for families given in explicit
  form y = f(x, C) or implicit form F(x,y,C)=0 (limited implicit support).
- Numerically plots the original family and its orthogonal trajectories.
- Several built-in example families with ready-to-run plots.
- Command-line interface to choose examples or provide your own expression.

Requirements
- Python 3.8+
- sympy
- numpy
- matplotlib

Install via pip if needed:
    pip install sympy numpy matplotlib

How this works (short)
1. For an explicit family y = f(x,C), differentiate with respect to x to get y'.
2. Eliminate the parameter C (solve for C from the family relation and substitute).
3. Replace y' by -1 / y' to get the differential equation of orthogonal trajectories.
4. Solve the resulting ODE (symbolically if possible, otherwise integrates numerically
   or gives an implicit solution).

Usage examples (run from terminal):
    python orthogonal_trajectory.py --example 1
    python orthogonal_trajectory.py --example 2 --n_curves 10

Built-in examples
1) Concentric circles: x**2 + y**2 = C**2
   Orthogonal trajectories: straight lines through origin.
2) Parabolas: y = C*x**2
   Orthogonal trajectories: curves satisfying y**2 + x**2/2 = const
3) Family: y = C*exp(x)
   Orthogonal trajectories: solve and plot numerically if needed.

You can also import the functions from this module in your own scripts.

"""

from __future__ import annotations
import argparse
import math
from typing import Optional, Tuple

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- Core symbolic utilities ----------------------------

def orthogonal_trajectory_explicit(f_expr: sp.Expr, x: sp.Symbol, y: sp.Symbol, C: sp.Symbol) -> sp.Eq:
    """
    Compute a symbolic implicit equation for the orthogonal trajectories of the
    family y = f_expr(x, C).

    Returns a SymPy equation (implicit) representing the orthogonal trajectories.
    If SymPy can solve the ODE, the returned expression will be an explicit or
    implicit closed form. Otherwise it returns an implicit integral form.
    """
    # 1) Original family: y - f(x, C) = 0
    family_eq = sp.Eq(y, f_expr)

    # 2) Differentiate implicitly with respect to x: dy/dx = df/dx
    dy_dx = sp.diff(f_expr, x)

    # 3) Solve family equation for C (attempt)
    try:
        sol_for_C = sp.solve(sp.Eq(y, f_expr), C)
    except Exception:
        sol_for_C = []

    if sol_for_C:
        # Substitute the solved C into dy/dx to eliminate parameter
        C_sub = sol_for_C[0]
        dy_dx_sub = sp.simplify(dy_dx.subs(C, C_sub))
    else:
        # If we cannot isolate C, attempt to eliminate using eliminate
        # Create the relation and eliminate C
        rel = sp.Eq(y - f_expr, 0)
        try:
            # This returns a list of polynomials; then solve for dy/dx symbolically is hard.
            dy_dx_sub = dy_dx
        except Exception:
            dy_dx_sub = dy_dx

    # 4) Slope of orthogonal trajectories = -1 / (dy/dx)
    orth_slope = sp.simplify(-1 / dy_dx_sub)

    # 5) Form ODE: dy/dx = orth_slope
    ode = sp.Eq(sp.Derivative(y, x), orth_slope)

    # 6) Try to solve ODE
    try:
        sol = sp.dsolve(ode)
        # dsolve returns an Eq; try to return it
        return sol
    except Exception:
        # If dsolve fails, return implicit integral form: dy/orth_slope = dx
        # rearrange: orth_slope is a function of x and y
        # integral(1, dy/orth_slope) - integral(1, dx) = const
        # we'll return an Eq of the symbolic integrals (formal)
        Int_y = sp.integrate(1 / orth_slope, (y,))  # integrate w.r.t y symbolically
        Int_x = sp.integrate(1, (x,))  # this is just x
        return sp.Eq(Int_y - x, sp.Symbol('C1'))


# ---------------------------- Numeric plotting utilities ----------------------------

def plot_family_and_orthogonals(explicit_f: Optional[sp.Expr], implicit_F: Optional[sp.Expr],
                                x: sp.Symbol, y: sp.Symbol, C: sp.Symbol,
                                c_values: list[float], orth_solution: Optional[sp.Expr] = None,
                                xlim: Tuple[float, float] = (-5, 5), ylim: Tuple[float, float] = (-5, 5),
                                n_plot_points: int = 400, title: str = "Family + Orthogonal Trajectories") -> None:
    """
    Draws the family defined either explicitly (y = f(x, C)) or implicitly (F(x,y,C)=0)
    and overlays orthogonal trajectories if orth_solution is provided (as a SymPy Eq or expression).
    """
    xs = np.linspace(xlim[0], xlim[1], n_plot_points)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot family curves
    if explicit_f is not None:
        f_lamb = sp.lambdify((x, C), explicit_f, 'numpy')
        for c in c_values:
            ys = f_lamb(xs, c)
            # Mask invalid values
            mask = np.isfinite(ys)
            ax.plot(xs[mask], ys[mask], linewidth=1)
    elif implicit_F is not None:
        # For implicit curves, contour the function F(x,y,c)=0 for each c
        X, Y = np.meshgrid(xs, np.linspace(ylim[0], ylim[1], n_plot_points))
        F_lamb = sp.lambdify((x, y, C), implicit_F, 'numpy')
        for c in c_values:
            Z = F_lamb(X, Y, c)
            ax.contour(X, Y, Z, levels=[0], linewidths=1)

    # Plot orthogonal trajectories
    if orth_solution is not None:
        # orth_solution may be an Eq or list-like returned by dsolve
        sol_exprs = []
        if isinstance(orth_solution, sp.Equality):
            sol_exprs = [orth_solution]
        elif isinstance(orth_solution, list) or isinstance(orth_solution, tuple):
            sol_exprs = list(orth_solution)
        else:
            sol_exprs = [orth_solution]

        for sol in sol_exprs:
            # sol could be Eq(y(x), expression) or something similar
            try:
                if isinstance(sol, sp.Equality):
                    rhs = sol.rhs
                    # create lambdify for rhs where 'C1' is a parameter
                    C1 = sp.symbols('C1')
                    rhs_lamb = sp.lambdify((x, C1), rhs, 'numpy')
                    # sample a few C1 values
                    for k in np.linspace(-3, 3, 7):
                        try:
                            ys = rhs_lamb(xs, k)
                            mask = np.isfinite(ys)
                            ax.plot(xs[mask], ys[mask], linestyle='--', linewidth=1.2)
                        except Exception:
                            pass
                else:
                    # Fall back: try to interpret sol as implicit (e.g., y**2 + x**2/2 = C1)
                    # We'll attempt to isolate y if possible
                    C1 = sp.symbols('C1')
                    try:
                        sol_iso = sp.solve(sp.Eq(sol, C1), y)
                        for expr in sol_iso:
                            expr_lamb = sp.lambdify((x, C1), expr, 'numpy')
                            for k in np.linspace(-3, 3, 7):
                                ys = expr_lamb(xs, k)
                                mask = np.isfinite(ys)
                                ax.plot(xs[mask], ys[mask], linestyle='--', linewidth=1.2)
                    except Exception:
                        pass
            except Exception:
                continue

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


# ---------------------------- Example families ----------------------------

def example_concentric_circles() -> None:
    x, y, C = sp.symbols('x y C')
    F = x**2 + y**2 - C**2
    # family is implicit: x^2 + y^2 = C^2

    # For circles, orth trajectories are radial lines y = m x. Let's compute via method:
    # Implicit differentiation: 2x + 2y*y' = 0 -> y' = -x/y -> orth slope = -1 / y' = y/x
    orth_slope = y / x
    ode = sp.Eq(sp.Derivative(y, x), orth_slope)
    sol = sp.dsolve(ode)

    print("Symbolic orthogonal trajectories for concentric circles:")
    print(sol)

    plot_family_and_orthogonals(None, F, x, y, C, c_values=[0.5, 1, 1.5, 2, 2.5], orth_solution=sol,
                                xlim=(-3, 3), ylim=(-3, 3), title='Concentric circles and orthogonals')


def example_parabolas() -> None:
    x, y, C = sp.symbols('x y C')
    f = C * x**2  # y = C x^2
    sol = orthogonal_trajectory_explicit(f, x, y, C)

    print("Orthogonal trajectories for y = C*x^2 (symbolic):")
    print(sol)

    plot_family_and_orthogonals(f, None, x, y, C, c_values=[-1, -0.5, 0.5, 1, 1.5], orth_solution=sol,
                                xlim=(-4, 4), ylim=(-4, 4), title='Parabolas y=C*x^2 and orthogonals')


def example_exponential_family() -> None:
    x, y, C = sp.symbols('x y C')
    f = C * sp.exp(x)
    sol = orthogonal_trajectory_explicit(f, x, y, C)

    print("Orthogonal trajectories for y = C*e^x (symbolic/implicit):")
    print(sol)

    plot_family_and_orthogonals(f, None, x, y, C, c_values=[0.2, 0.5, 1, 2, 3], orth_solution=sol,
                                xlim=(-4, 4), ylim=(-2, 8), title='y=C*e^x and orthogonals')


# ---------------------------- Command-line interface ----------------------------

def main() -> None:
    p = argparse.ArgumentParser(description='Compute and plot orthogonal trajectories for example families.')
    p.add_argument('--example', type=int, default=1, help='Example to run: 1=circles,2=parabolas,3=exp')
    p.add_argument('--n_curves', type=int, default=7, help='Number of family curves to plot (if applicable)')
    args = p.parse_args()

    if args.example == 1:
        example_concentric_circles()
    elif args.example == 2:
        example_parabolas()
    elif args.example == 3:
        example_exponential_family()
    else:
        print('Unknown example. Choose 1, 2 or 3.')


if __name__ == '__main__':
    main()
