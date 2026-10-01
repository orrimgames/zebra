# Browser teleop port

The website ports `view.py --teleop`, not a video player pretending to be a simulator.

- Physics: official `@mujoco/mujoco` 3.14.0 single-threaded WASM (Apache-2.0).
- Model: `scene.xml` generated from `orb/model.py` and `orb/world.py`, payload 5 kg.
- Control: speed, lean steering and leveling equations ported from `orb/control.py` into `controller.mjs`.
- Graphics: Three.js rendering of the original model geoms, with double-sided duplicate triangles removed from render meshes to prevent z-fighting. This does not change collision/physics data.
- Touch controls: hold arrows to drive, release to brake. Keyboard arrows, Space and O also work.
- Stops on tab hiding; pause/resume and reset supported.

Differences from desktop teleop: hold-to-drive touch UI rather than incremental keypress setpoints; no camera sensor views yet; scripted moving agents have not been ported and stay at their initial positions. The native autonomous planner and noisy sensor pipeline are not part of this manual mode. No hardware connection.

Initial load is about 15 MB. Physics can run slower than wall time on weaker phones; simulated speed/distance are reported, never claimed as measured hardware performance.

Upstream license: https://github.com/google-deepmind/mujoco/blob/main/LICENSE
