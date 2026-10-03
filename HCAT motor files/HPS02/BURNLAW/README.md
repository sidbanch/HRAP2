# HPS02 preliminary with the burn-rate law (9/28, results updated 10/2)

Sid's N2665 (`../ROUND2/HPS02-6_ROUND2_4500N_ZIGG.json`) rerun with HRAP's burn-rate law instead of fixed O/F 6, at the grain lengths Sid said the rocket could take (15 to 24"). Sims by Callan in HRAP2.

## What changed from Sid's file

- **Fuel flow:** burn-rate law (Shifting OF, a = 0.198, n = 0.325) instead of fixed O/F 6. Constant O/F is stored as 7.
- **Grain:** 15", 18" or 24" long (was 12"), 3.39" OD (the 4" chamber's liner ID; was 3.625"), 2.0" port.
- **Throat:** keeps peak chamber pressure at or under 500 psi on an 80 °F day under both the burn-rate law and fixed O/F 7 (was 1.35").
- **Unchanged:** 5.5" ID x 32" tank at 90% fill and 70 °F, Ziggurat modeled as 1 x 0.4" at Cd 0.5 (0.063 in²), SPI injector model, C* efficiency 85%, expansion ratio 3.2, masses.

## Results, 70 °F

| File | Throat | O/F (liquid-burn avg) | Peak Pc | Peak thrust | Avg thrust | Total impulse | Liquid burn | Port at end |
|---|---|---|---|---|---|---|---|---|
| Sid's file (fixed O/F 6, 12") | 1.35" | 6 (forced) | 492 psi | 4626 N | 2665 N | 18357 N·s | | |
| HPS02-6_15in_burnlaw | 1.406" | 16.7 | 417 psi | 4189 N | 2388 N | 14755 N·s | 3.6 s | 2.74" |
| HPS02-6_18in_burnlaw | 1.408" | 13.7 | 427 psi | 4315 N | 2453 N | 15412 N·s | 3.6 s | 2.75" |
| HPS02-6_24in_burnlaw | 1.409" | 10.1 | 442 psi | 4503 N | 2531 N | 16453 N·s | 3.8 s | 2.78" |

- Peak chamber pressure at 80 °F: 441 to 470 psi with the burn-rate law.
- On 10/2 HRAP2's ABS table was extended from O/F 10 to 30. Before that, HRAP reused the O/F 10 values above O/F 10, which overstated the 15" and 18" files' impulse by 7% and 4%. The table above uses the extended table.
- With fixed O/F 7 and the same throats: about 4750 N peak and 17640 N·s.
- All stay N-class (10240 to 20480 N·s).

## If the grain burns 1.5x faster than the literature values

| Grain | O/F | Port at end |
|---|---|---|
| 15" | 10.6 | 3.12" |
| 18" | 8.7 | 3.15" |
| 24" | 6.4 | 3.23" |

- No burnout in any case (3.39" liner). Helical grains or swirl injection are the expected source of faster burning; neither is modeled.

## Not included

- a and n are HRAP's unsourced straight-bore ABS values.
- The Ziggurat's CdA is unmeasured (estimates range 0.03 to 0.06 in²).
- Masses, lengths and CG are Sid's, not updated for the longer grain.
