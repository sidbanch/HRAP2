# HRAP2 — HCAT's Hybrid Rocket Analysis Program

HRAP2 simulates a nitrous hybrid motor's burn: tank, injector, fuel grain, chamber and nozzle. It also sizes a motor from targets like chamber pressure and O/F. It's HCAT's Python fork of [HRAP](https://github.com/rnickel1/HRAP_Source). The default model reproduces the original MATLAB HRAP; everything else is opt-in.

Matching MATLAB checks the code, not the physics. Only a hot fire or cold flow says how close a result is to a real motor.

## Run the app

- **Windows:** download the zip from [Releases](https://github.com/sidbanch/HRAP2/releases), unzip it and run `HRAP.exe`. From a source checkout, double-click `run_hrap.bat`.
- **macOS:** double-click `run_hrap.command`. It sets up `.venv` and installs what's missing. The first run needs Python 3.10+ or `uv`.
- **Anything else:** in a Python 3.10+ virtual environment, `python -m pip install -e .`, then `hrap`.

**Save** and **Load** (top right) read and write motor files as JSON, including the Sizing page settings. MATLAB `.mat` motor files load too. **Settings → Units** picks display units (psi, in, …); it doesn't change the calculation. Pressures are absolute, except injector ΔP.

## Sizing tab

Holds the whole motor: tank, fuel, injector, grain and nozzle. It also works out the injector, throat and grain for start-of-burn targets, using the same injector, combustion and nozzle math as the simulation. **Apply to motor** makes the sized throat, expansion ratio, hole count, swirler holes and grain length the motor's.

Motor files that give the tank by length, starting pressure or oxidizer mass, or the nozzle by exit diameter, load as the equivalent volume, temperature, fill and expansion ratio.

**Injector: Size from** decides what sets the oxidizer flow:

| Size from | You give | It gives |
| --- | --- | --- |
| Hole count | hole or swirler count and geometry | the flow |
| Liquid burn time | how long the liquid should last | flow = liquid in the tank ÷ burn time |
| O/F | a starting O/F, with the grain on Grain length | the flow that gives that O/F |

Every mode shows **Total CdA**: the Cd × area the injector needs for that flow at this ΔP. Compare it with a cold-flow result. If the cold flow gives Cd on the exit area, Cd × exit area is the CdA.

**Grain: Size from** O/F gives a grain length; Grain length gives an O/F. **Nozzle** gives the throat and expansion ratio.

**Check across injector Cd** runs the full simulation for a range of throats and injector Cds, since the Cd is usually a guess until a cold flow. Cells are red over the chamber pressure limit and blue under the minimum. Click a throat to use it.

### Swirl injectors

Set the injector **Type** to **Swirler** for tangential-port swirlers. **From geometry** estimates the Cd from the exit, port count, port size and port offset with Abramovich's ideal swirl theory ([below](#how-the-swirl-model-works)).

In **Liquid burn time** or **O/F** mode, the page works backwards to a swirler instead:

- **Inlet port diameter:** the port size that gives the needed CdA, for the exit, port count and offset you enter.
- **Swirler layouts:** drillable options for a list of exits and port counts, rounded to real number drills, with the starting O/F and flow each one gives. Click a row to load it.

The theory ignores losses entering the ports, so real swirlers probably flow less than it says. Drill the listed size, cold-flow it, and open the ports up a drill size if it flows low.

## Simulation tab

Runs the motor from the Sizing page. The **Motor** line at the top says which motor that is, and lists anything the Sizing page calls for that isn't applied yet. Press **Run**, and pick what to plot from the list. Hover a plot to read the motor at that time.

The left side only holds how to simulate the motor:

- **Fuel flow:** **Burn-rate law** (HRAP's Shifting OF) burns the grain back at the fuel's burn rate, so the O/F drifts as the port opens. **Fixed O/F** (Constant OF) skips the grain and holds the O/F you type. Use it only without burn-rate data.
- **Run:** run time, when the valve closes, timestep, and chamber pressure before ignition.

Options that change the default MATLAB model:

- **Flow model** (Sizing page, Injector card):
  - **SPI** (default): pure liquid through the injector, as in original HRAP. Overpredicts flow above about 250–300 psi ΔP.
  - **HEM:** the nitrous boils instantly in the orifice. Underpredicts flow.
  - **Dyer:** a blend of the two, weighted by κ (default 1).

  HEM and Dyer use CoolProp nitrous properties for the injector and the tank.
- **Solve tank cooling each step:** replaces HRAP's averaged pressure drop near the end of the liquid with a calculated one. Changes total impulse by under 1%.
- **Enable advanced options:** live chemistry and experimental fluid and grain models.

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
| [`src/hrap/gui/`](src/hrap/gui/) | The desktop app: `main.py` (window, Simulation and Mass tabs), `sizing.py`, `sweep.py`, `swirler_options.py` |
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
