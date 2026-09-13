#!/usr/bin/env python3
"""Let RoboTwin's simulator start when curobo cannot be installed.

`envs/robot/robot.py` imports `CuroboPlanner` unconditionally and constructs one
per arm on every `reset()`. RoboTwin's own `planner.py` guards the curobo import
with a bare `try/except`, so on a box without curobo the import raises and the
simulator never starts.

The planner is not on the policy-evaluation path: `_base_task.take_action` routes
joint-space actions through the *mplib* TOPP planner (`robot.left_mplib_planner`),
and no task env touches the curobo planner. curobo's own build needs a CUDA 12.1
toolkit to match torch 2.4.1+cu121; this box only ships CUDA 13.2, so the
extension cannot compile here.

This script appends a stand-in class at the end of `planner.py` that is only
defined when the real import failed. Idempotent; pass --revert to undo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "# --- policy-eval fallback (added by scripts/patch_robotwin_curobo_fallback.py) ---"

BLOCK = f'''

{MARKER}
# Policy evaluation never queries the curobo planner: `_base_task.take_action`
# drives joint-space actions through the mplib TOPP planner. The object is only
# constructed (once per reset), so a stand-in keeps the simulator startable on a
# box whose CUDA toolkit cannot build curobo's extension.
if "CuroboPlanner" not in globals():

    class CuroboPlanner:  # noqa: D101
        def __init__(self, *args, **kwargs):
            self.disabled = True
            self.args = args
            self.kwargs = kwargs

        def __getattr__(self, name):
            raise AttributeError(
                f"curobo is not installed, so planner.{{name}} is unavailable; "
                "RoboTwin policy evaluation only uses the mplib TOPP planner"
            )
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path,
                        default=Path("/workspace/robotwin/code/envs/robot/planner.py"))
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    text = args.planner.read_text(encoding="utf-8")
    if args.revert:
        if MARKER not in text:
            print("nothing to revert")
            return
        args.planner.write_text(text.split("\n" + MARKER)[0] + "\n", encoding="utf-8")
        print(f"reverted {args.planner}")
        return

    if MARKER in text:
        print("already patched")
        return
    args.planner.write_text(text.rstrip("\n") + BLOCK, encoding="utf-8")
    print(f"patched {args.planner}")


if __name__ == "__main__":
    main()
