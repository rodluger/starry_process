import app
from line_profiler import LineProfiler
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


profile = LineProfiler()
profile.add_module(app)
profile.run("app.Application()")
profile.print_stats()
