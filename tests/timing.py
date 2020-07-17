from starry_gp.gp import YlmGP
from line_profiler import LineProfiler

gp = YlmGP(10)

profile = LineProfiler()
# profile.add_function(gp.set_params)
profile.add_function(gp.P.set_params)

profile.run("gp.set_params()")

profile.print_stats()
