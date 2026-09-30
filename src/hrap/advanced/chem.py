"""Optional live chemistry (advanced mode, not MATLAB-identical).

NumPy port of the Gibbs-energy minimizer in the legacy JAX ``chem.py``.
Default HRAP still uses frozen MATLAB ``.mat`` / JSON tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from hrap.engine.types import Propellant
from hrap.io.propellant import load_propellant

Rhat = 8314.0  # J/(K*kmol), same as the JAX ChemSolver
_ATM = 101325.0
N_CURVE_MAX = 3

# Product names to keep the live table builder interactive (full thermo.dat is huge).
DEFAULT_PRODUCTS = (
    "CO2", "CO", "H2O", "H2", "O2", "N2", "OH", "NO", "N2O", "NO2",
    "H", "O", "N", "C", "HO2", "H2O2", "NH3", "CH4",
)

FUEL_RECIPES = {
    "ABS": dict(formula="ABS", composition={"C": 8.0, "H": 8.0, "N": 1.0}, M=119.16, h0=147.0e6),
    "HDPE": dict(formula="HDPE", composition={"C": 2.0, "H": 4.0}, M=28.05, h0=-52.0e6),
    "HTPB": dict(formula="HTPB", composition={"C": 7.22, "H": 10.86, "O": 0.17}, M=100.0, h0=-12.0e6),
    "Paraffin": dict(formula="PARAFFIN", composition={"C": 32.0, "H": 66.0}, M=450.0, h0=-930.0e6),
    "HTPB_Paraffin": dict(formula="50P", composition={"C": 20.0, "H": 38.0, "O": 0.1}, M=280.0, h0=-400.0e6),
    "Asphalt": dict(formula="ASPHALT", composition={"C": 10.0, "H": 12.0, "S": 0.2}, M=140.0, h0=50.0e6),
    "Sorbitol": dict(formula="SORBITOL", composition={"C": 6.0, "H": 14.0, "O": 6.0}, M=182.17, h0=-1335.0e6),
    "Metalized_Plastisol": dict(formula="MPLAST", composition={"C": 4.0, "H": 6.0, "O": 1.0, "AL": 1.0}, M=86.0, h0=-150.0e6),
}


@dataclass
class NASA9:
    T_min: float
    T_max: float
    DeltaHForm: float
    coeffs: np.ndarray

    def get_Cp_D(self, T: float) -> float:
        c = self.coeffs
        return c[0] / (T * T) + c[1] / T + c[2] + c[3] * T + c[4] * T * T + c[5] * T ** 3 + c[6] * T ** 4

    def get_H_D(self, T: float) -> float:
        c = self.coeffs
        return (
            -c[0] / (T * T)
            + c[1] / T * np.log(T)
            + c[2]
            + c[3] * T / 2.0
            + c[4] * T * T / 3.0
            + c[5] * T ** 3 / 4.0
            + c[6] * T ** 4 / 5.0
            + c[7] / T
        )

    def get_S_D(self, T: float) -> float:
        c = self.coeffs
        return (
            -c[0] / (2.0 * T * T)
            - c[1] / T
            + c[2] * np.log(T)
            + c[3] * T
            + c[4] * T * T / 2.0
            + c[5] * T ** 3 / 3.0
            + c[6] * T ** 4 / 4.0
            + c[8]
        )


@dataclass
class ThermoSubstance:
    formula: str
    comment: str
    condensed: bool
    is_product: bool
    composition: Dict[str, float]
    M: float
    providers: list
    T_min: float
    T_max: float

    def get_prov(self, T: float) -> NASA9:
        i = 0
        for j in range(1, len(self.providers)):
            if T > self.providers[j].T_min:
                i = j
        return self.providers[i]

    def get_H_D(self, T: float) -> float:
        return self.get_prov(T).get_H_D(T)


def make_basic_reactant(formula: str, composition: dict, M: float, T0: float, h0: float, condensed=True) -> ThermoSubstance:
    """h0 is J/kmol at T0."""
    coeffs = np.zeros(9)
    coeffs[2] = h0 / Rhat / T0
    return ThermoSubstance(
        formula,
        "",
        condensed,
        False,
        {k.upper(): float(v) for k, v in composition.items()},
        M,
        [NASA9(T0, T0, h0, coeffs)],
        T0,
        T0,
    )


def _thermo_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "thermo.dat"


def _eval_curve(coeffs: np.ndarray, T: float, kind: str) -> float:
    c = coeffs
    if kind == "Cp":
        return c[0] / (T * T) + c[1] / T + c[2] + c[3] * T + c[4] * T * T + c[5] * T ** 3 + c[6] * T ** 4
    if kind == "H":
        return (
            -c[0] / (T * T)
            + c[1] / T * np.log(T)
            + c[2]
            + c[3] * T / 2.0
            + c[4] * T * T / 3.0
            + c[5] * T ** 3 / 4.0
            + c[6] * T ** 4 / 5.0
            + c[7] / T
        )
    return (
        -c[0] / (2.0 * T * T)
        - c[1] / T
        + c[2] * np.log(T)
        + c[3] * T
        + c[4] * T * T / 2.0
        + c[5] * T ** 3 / 3.0
        + c[6] * T ** 4 / 4.0
        + c[8]
    )


def _props_at(T_bounds: np.ndarray, coeffs: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = coeffs.shape[0]
    Cp = np.empty(n)
    H = np.empty(n)
    S = np.empty(n)
    for j in range(n):
        i = 0
        for k in range(T_bounds.shape[1]):
            if T_bounds[j, k, 1] > 0 and T > T_bounds[j, k, 0]:
                i = k
        Cp[j] = _eval_curve(coeffs[j, i], T, "Cp")
        H[j] = _eval_curve(coeffs[j, i], T, "H")
        S[j] = _eval_curve(coeffs[j, i], T, "S")
    return Cp, H, S


class ChemSolver:
    """HP Gibbs minimizer (Gordon–McBride reduced equations), NumPy implementation."""

    def __init__(self, chem_infos, product_allow=DEFAULT_PRODUCTS):
        self.substances: Dict[str, ThermoSubstance] = {}
        self.product_allow = set(product_allow) if product_allow else None
        if not isinstance(chem_infos, (list, tuple)):
            chem_infos = [chem_infos]
        for chem_info in chem_infos:
            if isinstance(chem_info, (str, Path)):
                for k, v in self.load_propep(chem_info).items():
                    self.substances.setdefault(k, v)
            elif isinstance(chem_info, ThermoSubstance):
                self.substances.setdefault(chem_info.formula, chem_info)

    def load_propep(self, chem_path) -> dict:
        substances = {}
        path = Path(chem_path)
        with path.open("r", encoding="utf-8", errors="replace") as chem_file:

            def readline():
                line = chem_file.readline().rstrip("\n")
                while line is not None and (len(line.strip(" ")) == 0 or (line and line[0] == "!")):
                    line = chem_file.readline().rstrip("\n")
                    if line == "":
                        return line
                return line

            line = readline()
            in_reactants = False
            while line:
                if line.startswith("thermo"):
                    chem_file.readline()
                elif line.startswith("END PRODUCTS"):
                    in_reactants = True
                elif line.startswith("END REACTANTS"):
                    break
                else:
                    formula = line[0:18].strip(" ")
                    comment = line[18:].strip(" ") if len(line) > 18 else ""
                    line = readline()
                    i = 0
                    fit_pieces = int(line[i : i + 2].strip(" ") or "0")
                    i += 10
                    composition = {}
                    for _ in range(5):
                        symbol = line[i : i + 2].strip(" ").upper()
                        i += 2
                        quantity = float(line[i : i + 6].strip(" ") or "0")
                        i += 6
                        if symbol:
                            composition[symbol] = quantity
                    phase = int(line[i : i + 2].strip(" ") or "0")
                    i += 2
                    condensed = phase != 0
                    M = float(line[i : i + 13].strip(" "))
                    i += 13
                    DeltaHForm = 1e3 * float(line[i : i + 15].strip(" "))
                    line = readline()
                    providers = []
                    if fit_pieces == 0:
                        T = float(line[0:11].strip(" "))
                        providers.append(NASA9(T, T, DeltaHForm, np.array([0.0] * 2 + [DeltaHForm / Rhat / T] + [0.0] * 5)))
                    else:
                        for j in range(fit_pieces):
                            i = 0
                            T_min = float(line[i : i + 11].strip(" "))
                            i += 11
                            T_max = float(line[i : i + 11].strip(" "))
                            i += 11
                            i += 1
                            i += 8 * 5
                            coeffs = np.zeros(9)
                            m = 0
                            line = readline()
                            ii = 0
                            for _ in range(5):
                                coeff = line[ii : ii + 16].strip(" ")
                                ii += 16
                                if coeff:
                                    coeffs[m] = float(coeff.replace("D", "E"))
                                    m += 1
                            line = readline()
                            ii = 0
                            for _ in range(2):
                                coeff = line[ii : ii + 16].strip(" ")
                                ii += 16
                                if coeff:
                                    coeffs[m] = float(coeff.replace("D", "E"))
                                    m += 1
                            ii += 16
                            for k in range(2):
                                coeff = line[ii : ii + 16].strip(" ")
                                ii += 16
                                if coeff:
                                    coeffs[7 + k] = float(coeff.replace("D", "E"))
                            providers.append(NASA9(T_min, T_max, DeltaHForm, coeffs))
                            if j != fit_pieces - 1:
                                line = readline()
                    T_min = min(p.T_min for p in providers)
                    T_max = max(p.T_max for p in providers)
                    substances[formula] = ThermoSubstance(
                        formula, comment, condensed, not in_reactants, composition, M, providers, T_min, T_max
                    )
                line = readline()
        return substances

    @dataclass
    class Result:
        T: float = 0.0
        Cp: float = 0.0
        Cv: float = 0.0
        gamma: float = 1.2
        M: float = 20.0
        R: float = 287.0
        valid: bool = False
        iters: int = 0

    def solve(self, Pc: float, supply: dict, max_iters: int = 80) -> Result:
        if not supply or Pc <= 0.0:
            return self.Result(valid=False)
        present_elements = ["E"]
        for formula in supply:
            sub = self.substances[formula]
            for elem in sub.composition:
                if elem not in present_elements:
                    present_elements.append(elem)
        present_elements = sorted(present_elements)

        gasses = []
        T_bounds = []
        coeffs = []
        for sub in self.substances.values():
            if not (sub.is_product and not sub.condensed):
                continue
            if self.product_allow and sub.formula not in self.product_allow:
                continue
            if not all(elem in present_elements for elem in sub.composition):
                continue
            n_curves = min(len(sub.providers), N_CURVE_MAX)
            tb = np.zeros((N_CURVE_MAX, 2))
            cf = np.zeros((N_CURVE_MAX, 9))
            for i, prov in enumerate(sub.providers[:n_curves]):
                tb[i] = [prov.T_min, prov.T_max]
                cf[i] = prov.coeffs
            T_bounds.append(tb)
            coeffs.append(cf)
            gasses.append(sub)
        if not gasses:
            return self.Result(valid=False)

        N_gas = len(gasses)
        N_elem = len(present_elements)
        gas_a = np.zeros((N_gas, N_elem))
        for i, gas in enumerate(gasses):
            for elem, amount in gas.composition.items():
                gas_a[i, present_elements.index(elem)] = amount
        T_bounds = np.asarray(T_bounds)
        coeffs = np.asarray(coeffs)

        T = 3000.0
        n = 0.1
        n_j = np.ones(N_gas) * 0.1 / N_gas
        pi_i = np.zeros(N_elem)
        Deltaln_n = 0.0
        Deltaln_T = 0.0

        h_0 = 0.0
        b_i0 = np.zeros(N_elem)
        for formula, inputs in supply.items():
            sub = self.substances[formula]
            if isinstance(inputs, (tuple, list)):
                m_frac, T_in = float(inputs[0]), float(inputs[1])
            else:
                m_frac, T_in = float(inputs), sub.T_min
            n_in = m_frac / sub.M
            h_0 += n_in * sub.get_H_D(T_in) * Rhat * T_in
            for elem, amount in sub.composition.items():
                b_i0[present_elements.index(elem)] += amount * n_in
        b_i0_max = float(np.max(b_i0)) if np.max(b_i0) > 0 else 1.0

        valid = False
        it = 0
        Deltan_j = np.zeros(N_gas)
        for it in range(1, max_iters + 1):
            gas_Cp_D, gas_H_D, gas_S_D = _props_at(T_bounds, coeffs, T)
            N_dof = N_elem + 2
            rhs = np.zeros(N_dof)
            n_j = np.clip(n_j, 1e-30, None)
            n = max(n, 1e-30)
            a_n = gas_a * n_j[:, None]
            mu = gas_H_D - gas_S_D + np.log(n_j / n) + np.log(Pc / 1e5)
            rhs[:N_elem] = -b_i0 + (a_n * pi_i[None, :]).sum(axis=1).sum() * 0  # filled below
            for k in range(N_elem):
                akn = a_n[:, k]
                rhs[k] = (
                    -b_i0[k]
                    + np.sum((akn[:, None] * gas_a) * pi_i[None, :])
                    + np.sum(akn * Deltaln_n)
                    + np.sum(akn * gas_H_D * Deltaln_T)
                    - np.sum(akn * mu)
                    + np.sum(akn)
                )
            rhs[-2] = (
                -n
                - n * Deltaln_n
                + np.sum((gas_a * n_j[:, None]) * pi_i[None, :])
                + np.sum(n_j * Deltaln_n)
                + np.sum(n_j * gas_H_D * Deltaln_T)
                + np.sum(n_j)
                - np.sum(n_j * mu)
            )
            rhs[-1] = (
                -h_0 / (Rhat * T)
                + np.sum((gas_a * (n_j * gas_H_D)[:, None]) * pi_i[None, :])
                + np.sum(n_j * gas_H_D * Deltaln_n)
                + np.sum(n_j * gas_H_D)
                + np.sum(n_j * (gas_Cp_D + gas_H_D * gas_H_D) * Deltaln_T)
                - np.sum(n_j * gas_H_D * mu)
            )
            jac = np.zeros((N_dof, N_dof))
            jac[:N_elem, :N_elem] = gas_a.T @ (gas_a * n_j[:, None])
            jac[:N_elem, -2] = np.sum(gas_a * n_j[:, None], axis=0)
            jac[:N_elem, -1] = np.sum(gas_a * (n_j * gas_H_D)[:, None], axis=0)
            jac[-2, :N_elem] = jac[:N_elem, -2]
            jac[-1, :N_elem] = jac[:N_elem, -1]
            jac[-2, -2] = np.sum(n_j) - n
            jac[-1, -2] = np.sum(n_j * gas_H_D)
            jac[-2, -1] = np.sum(n_j * gas_H_D)
            jac[-1, -1] = np.sum(n_j * (gas_Cp_D + gas_H_D ** 2))
            try:
                upd = np.linalg.solve(jac, rhs)
            except np.linalg.LinAlgError:
                break
            pi_i = pi_i - upd[:N_elem]
            Deltaln_n = Deltaln_n - upd[-2]
            Deltaln_T = Deltaln_T - upd[-1]
            Deltan_j = Deltaln_n + gas_H_D * Deltaln_T - mu + np.sum(gas_a * pi_i[None, :], axis=1)
            lambda1 = 5.0 * max(abs(Deltaln_T), abs(Deltaln_n), float(np.max(np.abs(Deltan_j))))
            ln_nj_n = np.log(n_j / n)
            v = np.abs((-ln_nj_n - 9.2103404) / np.where(np.abs(Deltan_j - Deltaln_n) < 1e-30, 1.0, Deltan_j - Deltaln_n))
            mask = (ln_nj_n <= -18.420681) & (Deltan_j >= 0.0)
            lambda2 = float(np.min(v[mask])) if np.any(mask) else np.inf
            lam = min(1.0, 2.0 / max(lambda1, 1e-12), lambda2)
            n_j = n_j * np.exp(lam * Deltan_j)
            n = n * np.exp(lam * Deltaln_n)
            T = float(T * np.exp(lam * Deltaln_T))
            T = float(np.clip(T, 200.0, 6000.0))
            sum_n = np.sum(n_j)
            mass_ok = np.all(np.abs(b_i0 - np.sum(gas_a * n_j[:, None], axis=0)) < b_i0_max * 1e-6)
            if (
                abs(Deltaln_T) <= 1e-4
                and (np.sum(n_j * np.abs(Deltan_j)) / max(sum_n, 1e-30)) <= 5e-6
                and (n * abs(Deltaln_n) / max(sum_n, 1e-30)) <= 5e-6
                and mass_ok
            ):
                valid = True
                break

        Cp_D, H_D, _S = _props_at(T_bounds, coeffs, T)
        Cp_frozen = float(np.sum(n_j * Cp_D) * Rhat)
        M = 1.0 / max(n, 1e-12)
        R = Rhat / M
        Cv = Cp_frozen - R
        gamma = Cp_frozen / Cv if Cv != 0 else 1.2
        return self.Result(T=T, Cp=Cp_frozen, Cv=Cv, gamma=gamma, M=M, R=R, valid=valid, iters=it)


def blend_tables(ident: str, scale_T: float = 1.0) -> Propellant:
    prop = load_propellant(ident)
    if scale_T != 1.0:
        prop.T = np.asarray(prop.T, dtype=float) * scale_T
    return prop


def build_of_pc_tables(base_ident: str = "ABS") -> Propellant:
    """MATLAB tables remain the high-fidelity default."""
    return load_propellant(base_ident)


def live_propellant_tables(base: str | Propellant, ox_formula: str = "N2O") -> Propellant:
    """Build k, M, T grids from the NumPy Gibbs solver. Not MATLAB-identical."""
    if isinstance(base, str):
        prop = load_propellant(base)
        ident = base
    else:
        prop = base
        ident = prop.name
    recipe = None
    for key, rec in FUEL_RECIPES.items():
        if key.lower() in ident.lower() or rec["formula"].lower() == ident.lower():
            recipe = rec
            break
    if recipe is None:
        recipe = FUEL_RECIPES["ABS"]
    fuel = make_basic_reactant(recipe["formula"], recipe["composition"], recipe["M"], 298.15, recipe["h0"])
    solver = ChemSolver([_thermo_path(), fuel])
    OF = np.asarray(prop.OF, dtype=float).ravel()
    Pc = np.asarray(prop.Pc, dtype=float).ravel()

    def _thin(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.size <= n:
            return arr
        idx = np.unique(np.round(np.linspace(0, arr.size - 1, n)).astype(int))
        return arr[idx]

    OF = _thin(OF, 8)
    Pc = _thin(Pc, 6)
    k = np.zeros((Pc.size, OF.size))
    M = np.zeros_like(k)
    T = np.zeros_like(k)
    for i, pc in enumerate(Pc):
        for j, of in enumerate(OF):
            m_f = 1.0 / (1.0 + max(of, 1e-6))
            m_o = 1.0 - m_f
            supply = {ox_formula: m_o, fuel.formula: m_f}
            try:
                res = solver.solve(float(pc), supply)
            except Exception:
                res = ChemSolver.Result(valid=False)
            if res.valid and np.isfinite(res.T):
                k[i, j] = float(np.clip(res.gamma, 1.05, 1.8))
                M[i, j] = float(np.clip(res.M, 8.0, 50.0))
                T[i, j] = float(np.clip(res.T, 500.0, 5500.0))
            else:
                k[i, j] = 1.25
                M[i, j] = 24.0
                T[i, j] = 2500.0
    return Propellant(
        name=f"{prop.name} (live chem)",
        opt_OF=prop.opt_OF,
        rho=prop.rho,
        reg=prop.reg,
        OF=OF,
        Pc=Pc,
        k=k,
        M=M,
        T=T,
    )
