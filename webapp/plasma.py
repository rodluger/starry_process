from bokeh.palettes import Plasma256
import numpy as np

__all__ = ["plasma"]

idx = np.array(np.linspace(0, 255, 101), dtype=int)
Plasma100 = np.array(Plasma256)[idx][::-1]
plasma = []
for v, hex_color in enumerate(Plasma100):
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    plasma.append("rgba({:d}, {:d}, {:d}, 1) {:d}%".format(r, g, b, v))
plasma = plasma[::3]
plasma = ",\n    ".join(plasma)
plasma = "background: linear-gradient(\n{:s}\n);".format(plasma)
