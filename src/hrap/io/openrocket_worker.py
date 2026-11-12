"""Flies motors on a rocket in OpenRocket's own simulator. Its own process, as it loads OpenRocket's Java:
``python -m hrap.io.openrocket_worker`` with a job on stdin (see hrap.io.openrocket for the job and the replies)."""
from __future__ import annotations

import json
import math
import sys

from hrap.io.openrocket import MARK


def say(**reply):
    print(MARK + json.dumps(reply), flush=True)


def _number(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _seed_wind(options):
    """Keep the gusts the same across motors and separate flights."""
    try:
        winds = [options.getAverageWindModel()]
        for level in options.getMultiLevelWindModel().getLevels():
            model = level.getClass().getDeclaredField("model")
            model.setAccessible(True)
            winds.append(model.get(level))
        for wind in winds:
            seed = wind.getClass().getDeclaredField("seed")
            seed.setAccessible(True)
            seed.setInt(wind, 0)
        options.setRandomSeed(0)
    except Exception:
        raise ValueError("Couldn't keep OpenRocket's gusts the same between flights. "
                         "This OpenRocket version's wind model isn't supported.") from None


def main():
    job = json.load(sys.stdin)
    import jpype

    jpype.startJVM(job["jvm"], "-Djava.awt.headless=true", "-Dapple.awt.UIElement=true",
                   # The rocket only needs its own parts, and the motors come from the job.
                   "-Dopenrocket.bypass.presets=true", "-Dopenrocket.bypass.motors=true", classpath=[job["jar"]])
    try:
        core = jpype.JPackage("info").openrocket.core
        java, guice = jpype.JPackage("java"), jpype.JPackage("com").google.inject

        @jpype.JImplements(guice.Module)
        class PresetBinding:  # OpenRocket's own window adds this; its simulator alone leaves it out
            @jpype.JOverride
            def configure(self, binder):
                binder.bind(core.database.ComponentPresetDao.class_).to(core.database.ComponentPresetDatabase.class_)

        module = core.startup.CoreModule()
        modules = guice.util.Modules.override(module, core.plugin.PluginModule()).with_(PresetBinding())
        core.startup.Application.setInjector(guice.Guice.createInjector(modules))
        module.startLoader()

        document = core.file.GeneralRocketLoader(java.io.File(job["ork"])).load()
        simulation = document.getSimulation(job["simulation"])
        config = simulation.getFlightConfigurationId()
        mounts = [c for c in simulation.getRocket() if isinstance(c, core.rocketcomponent.MotorMount) and c.isMotorMount()]
        if not mounts:
            raise ValueError("The rocket has no motor mount. In OpenRocket, tick \"This component is a motor mount\" "
                             "on its motor tube.")
        if len(mounts) != 1:
            raise ValueError(f"This rocket has {len(mounts)} motor mounts. "
                             "The Flight tab only handles a rocket with one for now.")
        _seed_wind(simulation.getOptions())
    except Exception as exc:  # nothing can be flown
        say(error=_message(exc))
        return
    types, launch_rod = core.simulation.FlightDataType, core.simulation.FlightEvent.Type.LAUNCHROD
    for i, file in enumerate(job["motors"]):
        try:
            stream = java.io.FileInputStream(file)
            try:
                motor = core.file.motor.GeneralMotorLoader().load(stream, java.io.File(file).getName()).get(0).build()
            finally:
                stream.close()
            mount = mounts[0]
            fitted = core.motor.MotorConfiguration(mount, config, mount.getMotorConfig(config))
            fitted.setMotor(motor)
            mount.setMotorConfig(fitted, config)
            simulation.simulate()
            data = simulation.getSimulatedData()
            branch = data.getBranch(0)
            t, altitude, speed, stability = ([float(v) for v in branch.get(kind)] for kind in (
                types.TYPE_TIME, types.TYPE_ALTITUDE, types.TYPE_VELOCITY_Z, types.TYPE_STABILITY))
            off_rail = next((float(e.getTime()) for e in branch.getEvents() if e.getType() == launch_rod), None)
            keep = [j for j, time in enumerate(t) if time <= 1.25 * float(data.getTimeToApogee())]
            say(motor=i, apogee=_number(data.getMaxAltitude()), max_speed=_number(data.getMaxVelocity()),
                max_mach=_number(data.getMaxMachNumber()), max_accel=_number(data.getMaxAcceleration()),
                rail_speed=_number(data.getLaunchRodVelocity()), time_to_apogee=_number(data.getTimeToApogee()),
                stability=next((s for time, s in zip(t, stability) if off_rail is not None and time >= off_rail
                                and math.isfinite(s)), None),
                warnings=[str(w) for w in simulation.getSimulatedWarnings()],
                t=[t[j] for j in keep], altitude=[altitude[j] for j in keep], speed=[speed[j] for j in keep])
        except Exception as exc:
            say(motor=i, error=_message(exc))


def _message(exc: Exception) -> str:
    """The first line of what went wrong, without Java's class names."""
    text = str(getattr(exc, "getMessage", lambda: None)() or exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:300]


if __name__ == "__main__":
    main()
