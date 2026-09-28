# HPS01-2, 24" grain (9/28)

Candidate HPS01-2 motors with a 24" straight-bore grain, one file per injector size. Sims by Callan in HRAP2.

- **Injector CdA** is relative to HPS01-1's four injectors: 1x = 4 x 0.19" at Cd 0.19 = 0.0215 in² (fire video fit). The HPS02-6 4500N injector (1 x 0.4" at Cd 0.5 = 0.0628 in²) is about 3x.
- **Throat:** keeps peak chamber pressure at or under 500 psi on an 80 °F day under both fuel models.
  - For 2x to 3x, fixed O/F 7 gives the higher pressure (Sid's 9/27 note: size the nozzle for O/F 7 to 8).
  - For 1x and 1.5x, the 24" grain makes the burn-rate law run richer than O/F 7, so the burn-rate law sets the throat (1x: 0.845" instead of 0.827").
- **Files are set to the burn-rate law** (Shifting OF, a = 0.198, n = 0.325), 70 °F. Constant O/F is stored as 7; switch Fuel flow to Fixed O/F to compare.

## Results, burn-rate law, 70 °F

| File | Injector CdA | Throat | O/F (liquid-burn avg) | Peak Pc | Peak thrust | Total impulse | Liquid burn | Fuel burned | Port at end |
|---|---|---|---|---|---|---|---|---|---|
| HPS01-2_24in_1x | 0.0215 in² (1x) | 0.845" | 4.8 | 467 psi | 1711 N | 9894 N·s | 5.8 s | 1.71 kg | 3.03" |
| HPS01-2_24in_1p5x | 0.0323 in² (1.5x) | 1.012" | 6.4 | 465 psi | 2447 N | 9505 N·s | 3.9 s | 1.19 kg | 2.76" |
| HPS01-2_24in_2x | 0.0430 in² (2x) | 1.164" | 7.9 | 455 psi | 3165 N | 9085 N·s | 2.9 s | 0.92 kg | 2.61" |
| HPS01-2_24in_2p5x | 0.0537 in² (2.5x) | 1.298" | 9.3 | 447 psi | 3864 N | 8786 N·s | 2.3 s | 0.76 kg | 2.51" |
| HPS01-2_24in_3x | 0.0645 in² (3x) | 1.418" | 10.6 | 445 psi | 4585 N | 8613 N·s | 1.9 s | 0.66 kg | 2.45" |

## Peak chamber pressure at 80 °F

| Injector | Burn-rate law | Fixed O/F 7 | Fixed O/F 6 |
|---|---|---|---|
| 1x | 499 psi | 487 psi | 496 psi |
| 1.5x | 495 psi | 498 psi | 507 psi |
| 2x to 3x | 475 to 484 psi | 499 psi | 507 psi |

- Fixed O/F 6 at 80 °F is 1.5% over 500 psi (507 psi) for the 1.5x to 3x throats, which follow Sid's O/F 7 to 8 guidance.

## Checks

- **Burnout:** the port grows to at most 3.03" (1x) against the 3.39" liner ID, so the grain doesn't burn through; every run ends with the tank empty.
- **Length vs O/F:** O/F x grain length stays about constant, so these scale to other lengths (see the Notion page "HPS01-2 Grain Length: Burn-Rate Law 9/28").

## Inputs

- Base: `HPS01-01_Massed.json` (HPS01-1's tank, 45" of 3.625" bore at 75% fill; C* efficiency 85%; nozzle efficiency 100%; expansion ratio 3.2; SPI injector model; 1 ms step).
- Grain: ABS, 2.0" port, 3.39" OD (liner ID), 24" long, straight bore.
- Injector: 4 holes of 0.19" with Cd scaled to the CdA (only the total CdA matters to the SPI model).
- Pressures are absolute.

## Not included

- a and n are HRAP's unsourced straight-bore ABS values; helical grains or swirl would lower the O/F by an unmeasured amount.
- HRAP's ABS combustion table stops at O/F 10, so the 3x row overstates thrust slightly.
- Pre- and post-combustion chamber lengths are HPS01-1's (4.39" each); motor masses and positions are HPS01-1's and not updated for the longer chamber.
