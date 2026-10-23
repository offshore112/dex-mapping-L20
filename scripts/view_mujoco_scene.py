"""Open an interactive MuJoCo viewer for an XML scene."""
from __future__ import annotations

import argparse
import time
from pathlib import Path


DEFAULT_SCENE = Path("experiment/v6_mujoco_static_import/mujoco_l20_mano_r_v6.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="MuJoCo XML path; defaults to the v6 L20 + MANO-R scene",
    )
    parser.add_argument(
        "--qpos",
        type=float,
        nargs="*",
        default=None,
        help="optional qpos values in the scene joint order",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="load the scene and print its dimensions without opening a window",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="advance MuJoCo physics; by default the viewer is frozen for inspection",
    )
    args = parser.parse_args()

    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"scene XML not found: {scene}")

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if args.qpos is not None:
        if len(args.qpos) != model.nq:
            raise ValueError(
                f"--qpos expects exactly {model.nq} values for this scene, "
                f"but received {len(args.qpos)}"
            )
        data.qpos[:] = args.qpos
        mujoco.mj_forward(model, data)

    print(f"scene: {scene}")
    print(f"bodies={model.nbody}, joints={model.njnt}, qpos={model.nq}, geoms={model.ngeom}")
    if args.check_only:
        return
    print("Close the MuJoCo window to return to the terminal.")
    if args.simulate:
        print("mode: physics simulation (joint states may move without controllers)")
        mujoco.viewer.launch(model, data)
    else:
        print("mode: frozen inspection (physics is not advanced)")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
