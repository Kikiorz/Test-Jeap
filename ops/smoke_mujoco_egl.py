#!/usr/bin/env python3
"""Render one small frame without nested Python/shell/XML quoting."""
import json
import os

SCENE_XML = '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'


def main():
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        frame = renderer.render()
        if frame.shape != (64, 64, 3) or not np.isfinite(frame).all():
            raise RuntimeError('Invalid EGL frame')
    finally:
        renderer.close()
    print(json.dumps({'passed': True, 'renderer': 'egl', 'shape': list(frame.shape)}), flush=True)


if __name__ == '__main__':
    main()
