"""Command line entry point.

    dronelab <task> <command> [options]
    python -m dronelab <task> <command> [options]

Tasks:     hover, cargo, mission
Commands:  train, evaluate, visualize (and mission site, which builds the project site)

Run ``dronelab <task> <command> --help`` for the options of each command.
"""
from __future__ import annotations

import importlib
import sys

TASKS = {
    "hover": "single planar drone holding a hover point",
    "cargo": "three 3D drones holding one cargo on ropes",
    "mission": "the cargo team flying through generated maps",
}
COMMANDS = ("train", "evaluate", "visualize")
EXTRA_COMMANDS = {"mission": ("site",)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in TASKS or argv[1] not in COMMANDS + EXTRA_COMMANDS.get(argv[0], ()):
        print(__doc__.strip() + "\n")
        for name, desc in TASKS.items():
            print(f"  {name:<8} {desc}")
        return 0 if argv[:1] in ([], ["-h"], ["--help"]) else 2
    module = importlib.import_module(f"dronelab.{argv[0]}.{argv[1]}")
    module.main(argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
