# HRAP2 — HCAT's Hybrid Rocket Analysis Program

HRAP2 simulates a nitrous hybrid motor's burn: tank, injector, fuel grain, chamber and nozzle. It also sizes a motor from targets like chamber pressure and O/F. It's HCAT's Python fork of [HRAP](https://github.com/rnickel1/HRAP_Source). The default model reproduces the original MATLAB HRAP; everything else is opt-in.

Matching MATLAB checks the code, not the physics. Only a hot fire or cold flow says how close a result is to a real motor.

## Install

HRAP installs as an app and updates itself from this repo's `main` branch. It needs no admin rights, and sets up its own Python.

- **macOS:** download and double-click [`packaging/install_mac.command`](packaging/install_mac.command), or run
  `curl -fsSL https://raw.githubusercontent.com/sidbanch/HRAP2/main/packaging/install_mac.command | bash`.
  HRAP.app goes in your Applications folder.
- **Windows:** right-click [`packaging/install_windows.ps1`](packaging/install_windows.ps1) → Run with PowerShell, or run
  `powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/sidbanch/HRAP2/main/packaging/install_windows.ps1 | iex"`.
  Shortcuts go on the Desktop and in the Start menu.

A few seconds after it opens, the app checks for a newer version; **Update available** in the status bar installs it, then restarts. **Settings → Check for updates…** checks now, and **Settings → Update branch…** follows another branch (for testing unmerged work). Running the installer again repairs an install.

Quitting keeps everything: open motors, their unsaved edits, the page you were on, the window size and the Study tab's runs come back next time, including after a crash or an update restart.

**From a source checkout** (for working on HRAP): `run_hrap.command` (macOS) or `run_hrap.bat` (Windows) sets up `.venv` and runs the checkout. These copies don't update themselves; use git.

Motor files are JSON, and hold the Motor tab's Sizing settings and targets too. MATLAB `.mat` motor files open too. Each open motor gets a tab at the top; **Open…** adds one, and **Save** writes the motor back to its file (**File → Save As…** for a new file). Switching tabs keeps each motor's unsaved edits (marked •) and its last run. ⌘} / ⌘{ (Ctrl+Tab on Windows) steps through them. **Settings → Units** picks display units (psi, in, …); it doesn't change the calculation. Pressures are absolute, except injector ΔP.

## Motor tab

Holds the whole motor: tank, fuel, injector, grain and nozzle, and shows its state at the start of the burn (flows, O/F, chamber pressure, thrust), using the same injector, combustion and nozzle math as the simulation.

The Injector, Grain and Nozzle cards each have a **Sizing** setting. **Manual**, the part is what you enter. Set **For** a target instead, the card works out the part that meets it, and **Apply to motor** makes the sized throat, expansion ratio, hole count, swirler holes and grain length the motor's. A motor file without saved Sizing settings opens with every part manual.

**Fuel: Fuel flow** is **Burn-rate law** (HRAP's Shifting OF), where the grain burns back at a × G^n and the O/F drifts as the port opens, or **Fixed O/F** (Constant OF), which holds the O/F you type and ignores the burn rate. Use Fixed O/F only without burn-rate data. With it, nothing can be sized for an O/F.

Motor files that give the tank by length, starting pressure or oxidizer mass, or the nozzle by exit diameter, load as the equivalent volume, temperature, fill and expansion ratio.

**Injector: Sizing** decides what sets the oxidizer flow:

| Sizing | You give | It gives |
| --- | --- | --- |
| Manual | hole or swirler count and geometry | the flow |
| For liquid burn time | how long the liquid should last | flow = liquid in the tank ÷ burn time |
| For O/F | a starting O/F, with the grain length you enter | the flow that gives that O/F |

Every mode shows **Total CdA**: the Cd × area the injector needs for that flow at this ΔP. Compare it with a cold-flow result. If the cold flow gives Cd on the exit area, Cd × exit area is the CdA.

**Grain: Sizing** For O/F gives a grain length; Manual gives an O/F. **Nozzle: Sizing** For chamber pressure gives the throat and expansion ratio for a chamber pressure target; Manual uses the nozzle's own and gives the chamber pressure.

The throat × injector Cd sweep is now on the **Study** tab.

### Swirl injectors

Set the injector **Type** to **Swirler** for tangential-port swirlers. **From geometry** estimates the Cd from the exit, port count, port size and port offset with Abramovich's ideal swirl theory ([below](#how-the-swirl-model-works)).

Sized **For liquid burn time** or **For O/F**, the page works backwards to a swirler instead:

- **Inlet port diameter:** the port size that gives the needed CdA, for the exit, port count and offset you enter.
- **Swirler layouts:** drillable options for a list of exits and port counts, rounded to real number drills, with the starting O/F and flow each one gives. Click a row to load it.

Two settings tune the theory to measurements:

- **Inlet loss (ξ):** pressure lost entering the swirler holes (Bazarov). 0 is the ideal theory, which flows the most, so holes sized with it come out small; a sharp drilled hole is about 1.4. It matters most for small holes, and a cold flow of the swirler fits it.
- **Stock PTC acts like:** with **Stock** ticked, the PTC bore is the clean hole a stock 1/4" PTC flows like. Its tube stop and collet restrict more than its 0.188" hex. The default, 0.118", is a very rough estimate, marked "(rough)" until you change it: it's the size that makes HPS01-1's four swirlers run out of liquid at 5.8 s in the fire video, from one video timing, a guessed 0.10" hole offset and assumed flow settings (Dyer κ 1, burn-rate law, tank cooling, and the tube's measured 3.655" bore; SPI and a fixed O/F 6 give 0.100"). A cold flow of a bare stock PTC replaces it.

**Measured CdA → Fit** sets one of them from one swirler's cold-flow CdA (water flow ÷ √(2 × 998 kg/m³ × ΔP)): with **Stock** ticked it solves what the PTC acts like, with a bored PTC it solves ξ. One measurement fits one of the two, so flow a bare stock PTC too before trusting both.

Until a cold flow fits them, drill the listed size and open the holes up a drill size if the part flows low.

## Simulation tab

Runs the motor from the Motor tab. The **Motor** line at the top says which motor that is, and lists anything the Motor tab's sizing calls for that isn't applied yet. Press **Run**, and pick what to plot from the list. Hover a plot to read the motor at that time.

The left side only holds how to run it: run time, when the valve closes, timestep, and chamber pressure before ignition.

Options that change the default MATLAB model:

- **Flow model** (Motor tab, Injector card):
  - **SPI** (default): pure liquid through the injector, as in original HRAP. Overpredicts flow above about 250–300 psi ΔP.
  - **HEM:** the nitrous boils instantly in the orifice. Underpredicts flow.
  - **Dyer:** a blend of the two, weighted by κ (default 1).

  HEM and Dyer use CoolProp nitrous properties for the injector and the tank.
- **Solve tank cooling each step** (Simulation tab, Run): replaces HRAP's averaged pressure drop near the end of the liquid with a calculated one. Changes total impulse by under 1%.
- **Enable advanced options:** live chemistry and experimental fluid and grain models.

## Study tab

- Pick **Throat × injector Cd**, **Grain length × total injector CdA**, **Grain length × burn rate a**, or choose your own one or two inputs.
- Enter comma-separated values (`12, 15, 18, 24`) or `start:end:count` (`12:24:5`). Each input has its own units. CdA is the total over all injectors, not a multiplier of an unnamed reference.
- Choose the current fuel model, either model individually, or both. **Fixed O/F** and **burn rate a** can only be varied with the model that actually uses them.
- Press **Run study**. It snapshots the applied Motor settings and Simulation settings, then runs each combination in parallel. Apply pending Motor sizing first. The throat stays fixed unless it is one of the varied inputs; it is not automatically resized to a pressure target.
- Change **Show** to compare pressure, O/F, thrust, impulse, fuel consumed, port diameter, or burn times. Red cells exceed the pressure limit or deplete the fuel; amber cells have other warnings (hover to read them). O/F outside the combustion table is flagged, but the underlying HRAP calculation is unchanged.
- The minimum peak pressure and SPI ΔP warning retain the old sweep's screening controls. After a complete study, the page lists column values meeting those limits without fuel depletion across every row. This is a comparison of simulated cases, not a hardware qualification.
- Select cells to overlay up to eight thrust, pressure, O/F, or port-diameter curves. Double-click one to open its exact inputs as a new, unsaved motor.
- The **Results** list keeps studies for this app session, including their original motor inputs, even when you edit or switch motors. **Save study** writes those inputs and ranges to JSON; **Load study** restores them for another run. Curves are not saved in that file. **Export CSV** writes all case summaries in the current display units and labels incomplete batches as partial.
- **Stop** finishes only cases already running; it queues no more. Closing the app waits for those workers too.

Study O/F is total oxidizer mass divided by fuel mass during the simulated liquid phase. If a run ends before liquid runout, the runout cell says **Not reached** and O/F covers only the part simulated. Fuel depletion checks the existing straight cylindrical port model. Selecting injector Cd or CdA overrides geometry-derived Cd and uses the same Cd for SPI and HEM. Changing grain length keeps dry masses unchanged.

## Mass & export tab

Tank and chamber dry masses and positions give the empty mass and CG, and the CG over the burn for RSE and ENG files. **File → Export** writes CSV, RSE (OpenRocket / RockSim) or ENG.

## Command line

```sh
python -m hrap.cli motor.json -o HRAP_output.csv          # one run
hrap-sweep motor.json --throat 0.3:0.5:5 --cd 0.4:0.9:6   # throat × Cd sweep to CSV
hrap-compare motor.json golden.csv                        # compare against a saved MATLAB trace
```

## How the swirl model works

The spin leaves an empty air core down the exit, so liquid only flows through a ring around it. More swirl makes a bigger core and a lower Cd:

```
A  = port offset × exit radius ÷ (port count × port radius²)    swirl number
A  = (1 − φ) · √2 ÷ φ^1.5                                         φ = share of the exit filled with liquid
Cd = φ · √(φ ÷ (2 − φ))                                           Cd on the exit area
```

| A | 0.5 | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- | --- |
| Cd | 0.60 | 0.44 | 0.29 | 0.18 | 0.10 |

Ports farther off the axis, a bigger exit, or less total port area mean more swirl. More or bigger ports mean less. The exit is the narrowest point after the ports: on HCAT's plug-and-push-to-connect injector, that's the push-to-connect fitting's bore.

## Code layout

| Folder | What's in it |
| --- | --- |
| [`src/hrap/gui/`](src/hrap/gui/) | The desktop app: `main.py` (window, Simulation and Mass tabs), `sizing.py`, `study.py`, `swirler_options.py` |
| [`src/hrap/engine/`](src/hrap/engine/) | Tank, injector, grain, combustion, nozzle and the burn loop (`sim.py`); sizing (`sizing.py`) and the swirl model (`swirl.py`) |
| [`src/hrap/io/`](src/hrap/io/) | Motor files, units, propellant data and exports |
| [`src/hrap/advanced/`](src/hrap/advanced/) | Opt-in chemistry, fluid, injector and grain models |
| [`src/hrap/resources/`](src/hrap/resources/) | Propellant tables and example motors |
| [`tests/`](tests/) | Tests and saved MATLAB traces |
| [`reference/`](reference/) | The original MATLAB and Python HRAP, for comparison only ([guide](reference/README.md)) |

Upstream's Python HRAP also installs a package called `hrap`, so don't install both in one environment.

## Develop

See [the development guide](docs/development.md) for setup, tests, regenerating MATLAB traces and releases.

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

## Origin and license

Based on [HRAP](https://github.com/rnickel1/HRAP_Source) by Robert Nickel for the University of Tennessee Rocket Engineering Team. [GNU GPL v3](LICENSE).
