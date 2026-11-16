# Experimental tank mass and energy model

In Simulation → Run, check **Conserve tank mass and energy (experimental)**.
Unchecked uses the original MATLAB-derived HRAP tank. The existing
`solve_tank_cooling` setting now selects this model; the former approximate
liquid-cooling solver has been removed. Existing motors saved with that option
checked will use the new calculation when rerun. Saved runs are also recomputed.

The tank is rigid, contains one pure fluid, and has no separate pressurant. It
exchanges no heat with its wall. Liquid and vapor share one equilibrium temperature
and pressure. CoolProp recovers the state from density and internal energy:

```text
dm/dt = -(flow to chamber + external vent flow)
dU/dt = -sum(each outlet mass flow * that outlet's specific enthalpy)
```

Mass is in kg, energy in J, mass flow in kg/s, and specific enthalpy in J/kg.
The main outlet supplies saturated liquid until bulk liquid runout, then vapor.
Any subsequent equilibrium condensate stays in the tank. This is an explicit
outlet assumption, not a prediction of entrainment or liquid collection near the
outlet. Traces below one part per million of remaining mass count as runout.
The runout timestamp records the outlet transition even when condensate later
appears. No mass or energy is reset at that transition.

Liquid injector flow still uses the selected SPI, HEM, or Dyer calculation. Vapor
outlets use real-fluid isentropic expansion, with mass flux limited at choking.
An internal vent supplies vapor to the chamber; an external vent discharges to
ambient. Both remove mass and the appropriate outlet enthalpy once. Substeps near
runout preserve the specified flow rates rather than clipping them. Closing the
valve partway through a timestep uses only its open duration.

CSV exports include `T_tnk` [K], `m_liq` [kg], `mdot_v` [kg/s, external vent], and,
for the equilibrium model, `U_tnk` [J]. Internal vent flow is included in `mdot_o`.
Internal energy uses CoolProp's reference state; changes and the energy balance
are the useful quantities, not the absolute number by itself.

## What the checks establish

The automated checks cover mass and outlet-energy conservation, the dilute-gas
choking limit, blocked/reversed pressure drops, valve closure, tank-only timestep
convergence, and motor/export integration. The original HRAP MATLAB comparison
checks remain separate. These establish numerical behavior and agreement with
reference calculations, not agreement with a real motor.

The November 16, 2026 HPS01-2 comparison is local and uncommitted under
`HCAT motor files/Development/HPS01-2/analysis/tank-energy-2026-11-16/`.
It snapshots the actual motor inputs and compares without refitting them.
PT1's location is unconfirmed; it is plotted as a gauge channel plus 1 atm.
There is no measured oxidizer mass history or chamber pressure in this dataset.
The new tank model alone does not explain the measured thrust tail. Wall heat,
liquid/vapor discharge at the outlet, injector behavior, and the existing chamber,
combustion, and nozzle assumptions still need investigation before attributing
the remaining discrepancy to any one cause.

Property interface: [CoolProp low-level API](https://coolprop.org/coolprop/LowLevelAPI.html).
