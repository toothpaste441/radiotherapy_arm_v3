# Record one complete CyberKnife-style simulation cycle

From the project root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python robot_simulation.py --cyberknife-record
```

This records **one run of seven viewpoints**, from home through final hold,
without opening the interactive viewer. Output:

`results/cyberknife_motion/one_cycle.mp4`

Also writes the usual `multinode_summary.json`, samples CSV, and plots.
The MP4 is 960x540 at 20 frames per second; playback timing is based on
simulation time, not wall-clock computation time. Encoding requires FFmpeg,
installed automatically via `imageio-ffmpeg` in `requirements.txt`.

For the normal live viewer (without video recording):

```powershell
python robot_simulation.py --cyberknife-view
```

The recording records one run only; it does not loop endlessly.
For a **seamlessly repeating** clip, the robot would also need to return from
the last viewpoint to the starting pose, which is NOT implemented here.

If offscreen OpenGL cannot initialize on Windows, you can use OBS Studio to
capture the `--cyberknife-view` viewer instead. The recorder leaves all original
physics, controller, planner and alignment checks unchanged.
