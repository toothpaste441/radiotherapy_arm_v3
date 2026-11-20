"""Stream one MuJoCo run directly to an MP4 without screen recording.

Only the rendered simulation is recorded; stdout metrics are still written by
cyberknife_motion_demo.py.  Frame rate is based on *simulation time* so the video
is deterministic even if the simulation computes slower than real time.
"""
from __future__ import annotations

from pathlib import Path


class MuJoCoMP4Recorder:
    def __init__(self, model, output_path: Path, *, fps=20, width=960,
                 height=540, lookat=None, distance=3.0, azimuth=130., elevation=-22.):
        import mujoco
        import imageio.v2 as imageio
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('FPS and recording dimensions must be positive')
        self.model = model
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.frame_count = 0
        self._last_frame_index = -1
        # MuJoCo Renderer requires offscreen buffer dimensions <= offwidth/offheight.
        # The XML contains 1920x1080, sufficient for default settings.
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if lookat is not None:
            camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        self.camera = camera
        try:
            self.writer = imageio.get_writer(
                str(self.output_path), format='FFMPEG', mode='I',
                fps=self.fps, codec='libx264', quality=7,
                macro_block_size=16,
            )
        except BaseException:
            self.renderer.close()
            raise

    def capture(self, data):
        """Record at most one frame per simulation-time slot."""
        idx = int(round(float(data.time) * self.fps))
        if idx <= self._last_frame_index:
            return
        self.renderer.update_scene(data, camera=self.camera)
        image = self.renderer.render().copy()
        self.writer.append_data(image)
        self._last_frame_index = idx
        self.frame_count += 1

    def close(self):
        try:
            self.writer.close()
        finally:
            self.renderer.close()
