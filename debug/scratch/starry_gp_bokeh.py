import numpy as np
import starry
from starry_gp import YlmGP
from scipy.special import gamma
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Slider, Div
from bokeh.plotting import figure, output_file, show
from bokeh.models.tickers import FixedTicker
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper

# Settings
np.random.seed(0)
output_file("bokeh_test.html")
ydeg = 15
Ny = 50
nmaps = 5
xi_mu_0 = -0.1
xi_nu_0 = 0.01
phi_mu_0 = 0.90
phi_nu_0 = 0.03
r_mu_0 = 0.001
r_nu_0 = 0.1

# Custom CSS
mu = Div(
    text="""
<img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgoKPHN2ZwogICB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iCiAgIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiCiAgIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIKICAgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzb2RpcG9kaT0iaHR0cDovL3NvZGlwb2RpLnNvdXJjZWZvcmdlLm5ldC9EVEQvc29kaXBvZGktMC5kdGQiCiAgIHhtbG5zOmlua3NjYXBlPSJodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy9uYW1lc3BhY2VzL2lua3NjYXBlIgogICB3aWR0aD0iNDAwIgogICBoZWlnaHQ9IjQwMCIKICAgaWQ9InN2ZzQ2MTEiCiAgIHNvZGlwb2RpOnZlcnNpb249IjAuMzIiCiAgIGlua3NjYXBlOnZlcnNpb249IjAuOTIuNCAoNWRhNjg5YzMxMywgMjAxOS0wMS0xNCkiCiAgIHZlcnNpb249IjEuMCIKICAgc29kaXBvZGk6ZG9jbmFtZT0iR3JlZWtfbGNfbXUuc3ZnIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNDYxMyIgLz4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgaWQ9ImJhc2UiCiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEuMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMDAwMCIKICAgICBndWlkZXRvbGVyYW5jZT0iMTAiCiAgICAgb2JqZWN0dG9sZXJhbmNlPSIxLjYiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAuMCIKICAgICBpbmtzY2FwZTpwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnpvb209IjAuNyIKICAgICBpbmtzY2FwZTpjeD0iODUuOTA5MTkiCiAgICAgaW5rc2NhcGU6Y3k9IjE5MC42MTg4MyIKICAgICBpbmtzY2FwZTpkb2N1bWVudC11bml0cz0icHgiCiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0ibGF5ZXIxIgogICAgIGlua3NjYXBlOm9iamVjdC1iYm94PSJ0cnVlIgogICAgIGlua3NjYXBlOm9iamVjdC1wb2ludHM9InRydWUiCiAgICAgaW5rc2NhcGU6b2JqZWN0LW5vZGVzPSJ0cnVlIgogICAgIGlua3NjYXBlOmdyaWQtcG9pbnRzPSJ0cnVlIgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTkyMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctaGVpZ2h0PSIxMDE3IgogICAgIGlua3NjYXBlOndpbmRvdy14PSItOCIKICAgICBpbmtzY2FwZTp3aW5kb3cteT0iLTgiCiAgICAgd2lkdGg9IjQwMHB4IgogICAgIGhlaWdodD0iNDAwcHgiCiAgICAgc2hvd2dyaWQ9ImZhbHNlIgogICAgIGlua3NjYXBlOndpbmRvdy1tYXhpbWl6ZWQ9IjEiIC8+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhNDYxNiI+CiAgICA8cmRmOlJERj4KICAgICAgPGNjOldvcmsKICAgICAgICAgcmRmOmFib3V0PSIiPgogICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2Uvc3ZnK3htbDwvZGM6Zm9ybWF0PgogICAgICAgIDxkYzp0eXBlCiAgICAgICAgICAgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIgLz4KICAgICAgPC9jYzpXb3JrPgogICAgPC9yZGY6UkRGPgogIDwvbWV0YWRhdGE+CiAgPGcKICAgICBpbmtzY2FwZTpsYWJlbD0iTGF5ZXIgMSIKICAgICBpbmtzY2FwZTpncm91cG1vZGU9ImxheWVyIgogICAgIGlkPSJsYXllcjEiCiAgICAgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI1Ni42Nzk4LC01MzEuNzk2MykiPgogICAgPGcKICAgICAgIGFyaWEtbGFiZWw9Is68IgogICAgICAgc3R5bGU9ImZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0Om5vcm1hbDtsaW5lLWhlaWdodDowJTtmb250LWZhbWlseTonVGltZXMgTmV3IFJvbWFuJzt0ZXh0LWFsaWduOnN0YXJ0O3RleHQtYW5jaG9yOnN0YXJ0O2ZpbGw6IzAwMDAwMDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MXB4O3N0cm9rZS1saW5lY2FwOmJ1dHQ7c3Ryb2tlLWxpbmVqb2luOm1pdGVyO3N0cm9rZS1vcGFjaXR5OjEiCiAgICAgICBpZD0idGV4dDU1NDAiPgogICAgICA8cGF0aAogICAgICAgICBkPSJtIDQ4NS43MzExNCw4MDcuOTQ1OCBxIC0zNS4zNTE1NiwzNy44OTA2MyAtNjEuNTIzNDQsMzcuODkwNjMgLTE2Ljc5Njg3LDAgLTMxLjI1LC0xMi44OTA2MyB2IDEwLjM1MTU2IHEgMCwxNy45Njg3NSA1LjQ2ODc1LDM2LjEzMjgyIDYuMjUsMjAuMTE3MTggNi4yNSwyNi41NjI1IDAsOC45ODQzNyAtNS42NjQwNiwxNC44NDM3NSAtNS42NjQwNiw1Ljg1OTM3IC0xNC4wNjI1LDUuODU5MzcgLTguNTkzNzUsMCAtMTMuNjcxODcsLTYuODM1OTQgLTUuMDc4MTMsLTYuODM1OTMgLTUuMDc4MTMsLTE2LjQwNjI1IDAsLTcuMDMxMjUgNS4wNzgxMywtMjUuMzkwNjIgNS44NTkzNywtMjAuMzEyNSA1Ljg1OTM3LC0zOS40NTMxMyBWIDY2MS40NjE0MyBoIDMyLjIyNjU2IHYgMTE1LjgyMDMxIHEgMCwxNy41NzgxMiAyLjkyOTY5LDI1Ljc4MTI1IDMuMTI1LDguMjAzMTIgMTAuNzQyMTksMTMuNDc2NTYgNy44MTI1LDUuMDc4MTMgMTcuNTc4MTIsNS4wNzgxMyAxNy4zODI4MiwwIDQ1LjExNzE5LC0yMy44MjgxMyBWIDY2MS40NjE0MyBoIDMyLjQyMTg4IHYgMTM1Ljc0MjE4IHEgMCwxNy4xODc1IDMuNTE1NjIsMjMuODI4MTMgMy41MTU2Myw2LjQ0NTMxIDExLjkxNDA2LDYuNDQ1MzEgMTMuMjgxMjUsMCAxNy4xODc1LC0yNS45NzY1NiBoIDcuMDMxMjUgcSAtMy43MTA5Myw0NC4zMzU5NCAtMzcuNSw0NC4zMzU5NCAtMTQuNjQ4NDMsMCAtMjQuNDE0MDYsLTkuNzY1NjMgLTkuNTcwMzEsLTkuOTYwOTQgLTEwLjE1NjI1LC0yOC4xMjUgeiIKICAgICAgICAgc3R5bGU9ImZvbnQtc2l6ZTo0MDBweDtsaW5lLWhlaWdodDoxLjI1O3RleHQtYWxpZ246Y2VudGVyO3RleHQtYW5jaG9yOm1pZGRsZSIKICAgICAgICAgaWQ9InBhdGg4MTQiIC8+CiAgICA8L2c+CiAgPC9nPgo8L3N2Zz4K"
width=20, height=20></img>
""",
    css_classes=["custom-slider-title"],
)
nu = Div(
    text="""
<img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgoKPHN2ZwogICB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iCiAgIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiCiAgIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIKICAgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzb2RpcG9kaT0iaHR0cDovL3NvZGlwb2RpLnNvdXJjZWZvcmdlLm5ldC9EVEQvc29kaXBvZGktMC5kdGQiCiAgIHhtbG5zOmlua3NjYXBlPSJodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy9uYW1lc3BhY2VzL2lua3NjYXBlIgogICB3aWR0aD0iNDAwIgogICBoZWlnaHQ9IjQwMCIKICAgaWQ9InN2ZzQ2MTEiCiAgIHNvZGlwb2RpOnZlcnNpb249IjAuMzIiCiAgIGlua3NjYXBlOnZlcnNpb249IjAuOTIuNCAoNWRhNjg5YzMxMywgMjAxOS0wMS0xNCkiCiAgIHZlcnNpb249IjEuMCIKICAgc29kaXBvZGk6ZG9jbmFtZT0iR3JlZWtfbGNfbnUuc3ZnIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNDYxMyIgLz4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgaWQ9ImJhc2UiCiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEuMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMDAwMCIKICAgICBndWlkZXRvbGVyYW5jZT0iMTAiCiAgICAgb2JqZWN0dG9sZXJhbmNlPSIxLjYiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAuMCIKICAgICBpbmtzY2FwZTpwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnpvb209IjAuNyIKICAgICBpbmtzY2FwZTpjeD0iODUuOTA5MTkiCiAgICAgaW5rc2NhcGU6Y3k9IjE5MC42MTg4MyIKICAgICBpbmtzY2FwZTpkb2N1bWVudC11bml0cz0icHgiCiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0ibGF5ZXIxIgogICAgIGlua3NjYXBlOm9iamVjdC1iYm94PSJ0cnVlIgogICAgIGlua3NjYXBlOm9iamVjdC1wb2ludHM9InRydWUiCiAgICAgaW5rc2NhcGU6b2JqZWN0LW5vZGVzPSJ0cnVlIgogICAgIGlua3NjYXBlOmdyaWQtcG9pbnRzPSJ0cnVlIgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTkyMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctaGVpZ2h0PSIxMDE3IgogICAgIGlua3NjYXBlOndpbmRvdy14PSItOCIKICAgICBpbmtzY2FwZTp3aW5kb3cteT0iLTgiCiAgICAgd2lkdGg9IjQwMHB4IgogICAgIGhlaWdodD0iNDAwcHgiCiAgICAgc2hvd2dyaWQ9ImZhbHNlIgogICAgIGlua3NjYXBlOndpbmRvdy1tYXhpbWl6ZWQ9IjEiIC8+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhNDYxNiI+CiAgICA8cmRmOlJERj4KICAgICAgPGNjOldvcmsKICAgICAgICAgcmRmOmFib3V0PSIiPgogICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2Uvc3ZnK3htbDwvZGM6Zm9ybWF0PgogICAgICAgIDxkYzp0eXBlCiAgICAgICAgICAgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIgLz4KICAgICAgPC9jYzpXb3JrPgogICAgPC9yZGY6UkRGPgogIDwvbWV0YWRhdGE+CiAgPGcKICAgICBpbmtzY2FwZTpsYWJlbD0iTGF5ZXIgMSIKICAgICBpbmtzY2FwZTpncm91cG1vZGU9ImxheWVyIgogICAgIGlkPSJsYXllcjEiCiAgICAgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI1Ni42Nzk4LC01MzEuNzk2MykiPgogICAgPGcKICAgICAgIGFyaWEtbGFiZWw9Is69IgogICAgICAgc3R5bGU9ImZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0Om5vcm1hbDtsaW5lLWhlaWdodDowJTtmb250LWZhbWlseTonVGltZXMgTmV3IFJvbWFuJzt0ZXh0LWFsaWduOnN0YXJ0O3RleHQtYW5jaG9yOnN0YXJ0O2ZpbGw6IzAwMDAwMDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MXB4O3N0cm9rZS1saW5lY2FwOmJ1dHQ7c3Ryb2tlLWxpbmVqb2luOm1pdGVyO3N0cm9rZS1vcGFjaXR5OjEiCiAgICAgICBpZD0idGV4dDU1NDAiPgogICAgICA8cGF0aAogICAgICAgICBkPSJtIDQ0MS4wMDQ1OCw4NDUuODM2NDMgLTQzLjE2NDA2LC0xMjguMTI1IHEgLTQuMjk2ODgsLTEyLjY5NTMyIC0xMS4zMjgxMywtMjUgLTcuMDMxMjUsLTEyLjMwNDY5IC0xOS4zMzU5NCwtMTIuMzA0NjkgLTUuODU5MzcsMCAtMTIuODkwNjIsMi41MzkwNiBsIC0yLjczNDM4LC02LjgzNTk0IDQ4LjYzMjgyLC0xOS45MjE4NyBoIDguNTkzNzUgbCA1MS4zNjcxOCwxNTAuMTk1MzEgMzEuODM1OTQsLTY1LjAzOTA2IHEgOC43ODkwNiwtMTcuNTc4MTMgOC43ODkwNiwtMjcuNTM5MDYgMCwtOC4wMDc4MiAtMy4zMjAzMSwtMjIuNDYwOTQgLTIuNTM5MDYsLTEwLjU0Njg4IC0yLjUzOTA2LC0xNS44MjAzMSAwLC0xOS4zMzU5NCAyMC44OTg0NCwtMTkuMzM1OTQgMTguNTU0NjgsMCAxOC41NTQ2OCwxNy41NzgxMiAwLDE3LjM4MjgyIC0yNS43ODEyNSw2Ni40MDYyNSBsIC01NS44NTkzNywxMDUuNjY0MDcgeiIKICAgICAgICAgc3R5bGU9ImZvbnQtc2l6ZTo0MDBweDtsaW5lLWhlaWdodDoxLjI1O3RleHQtYWxpZ246Y2VudGVyO3RleHQtYW5jaG9yOm1pZGRsZSIKICAgICAgICAgaWQ9InBhdGg4MTQiIC8+CiAgICA8L2c+CiAgPC9nPgo8L3N2Zz4K"
width=20, height=20></img>
""",
    css_classes=["custom-slider-title"],
)
style_div = Div(
    text="""
<style>
    .custom-slider {
        left: 5px !important;
    }
    .custom-slider .bk-slider-title {
        margin-left: -3px;
    }
    .custom-slider .bk-slider-value {
        font-weight: unset;
    }
    .custom-slider-title {
        position: relative !important;
        text-align: right;
        width: 20px !important;
        height: 20px;
    }
</style>
"""
)
DisableTickLabels = FuncTickFormatter(code="return '  ';")

# Get the intensity design matrix `A`
Nx = 2 * Ny
x, y = np.meshgrid(
    2 * np.sqrt(2) * np.linspace(-1, 1, Nx), np.sqrt(2) * np.linspace(-1, 1, Ny)
)
a = np.sqrt(2)
b = 2 * np.sqrt(2)
idx = (y / a) ** 2 + (x / b) ** 2 > 1
y[idx] = np.nan
x[idx] = np.nan
theta = np.arcsin(y / np.sqrt(2))
lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)
lon = np.pi * x / (2 * np.sqrt(2) * np.cos(theta))
lat = lat.flatten() * 180 / np.pi
lon = lon.flatten() * 180 / np.pi
map = starry.Map(ydeg, lazy=False)
A = map.intensity_design_matrix(lat=lat, lon=lon)

# Draw three samples from the default distr
gp = YlmGP(ydeg)
gp.size.set_params(r_mu_0, r_nu_0)
gp.latitude.set_params(phi_mu_0, phi_nu_0)
gp.contrast.set_params(xi_mu_0, xi_nu_0)
ylm_0 = gp.draw(nmaps)

# Plot the GP ylm samples
ylm_plot = [None for i in range(nmaps)]
ylm_source = [None for i in range(nmaps)]
eps = 0.1
for i in range(nmaps):
    ylm_plot[i] = figure(
        plot_width=1400 // nmaps,
        plot_height=700 // nmaps,
        toolbar_location=None,
        x_range=(-2 - eps, 2 + eps),
        y_range=(-1 - eps / 2, 1 + eps / 2),
    )
    ylm_plot[i].axis.visible = False
    ylm_plot[i].grid.visible = False
    ylm_plot[i].outline_line_color = None
    foo = ylm_plot[i].image(
        image=[(A @ ylm_0[i]).reshape(Ny, Nx)],
        x=-2,
        y=-1,
        dw=4,
        dh=2,
        color_mapper=LinearColorMapper(palette="Viridis256", nan_color="white"),
    )
    ylm_source[i] = ylm_plot[i].renderers[0].data_source

# Now draw samples for random parameter combos
ncomb = 100
mu_r = np.empty(ncomb)
nu_r = np.empty(ncomb)
mu_phi = np.empty(ncomb)
nu_phi = np.empty(ncomb)
mu_xi = np.empty(ncomb)
nu_xi = np.empty(ncomb)
ylm = np.empty((ncomb, nmaps, (ydeg + 1) ** 2))
for i in range(ncomb):
    while True:
        try:
            mu_r[i] = np.random.random()
            nu_r[i] = np.random.random()
            mu_phi[i] = np.random.random()
            nu_phi[i] = np.random.random()
            mu_xi[i] = 2 * np.random.random() - 1
            nu_xi[i] = np.random.random()
            gp.size.set_params(mu_r[i], nu_r[i])
            gp.latitude.set_params(mu_phi[i], nu_phi[i])
            gp.contrast.set_params(mu_xi[i], nu_xi[i])
            ylm[i] = gp.draw(nmaps)
            break
        except:
            pass

# Collect all plots
ylm_layout = row(*ylm_plot, sizing_mode="stretch_width", margin=(10, 30, 10, 30))

# Contrast distribution
xi = np.linspace(-1, 1, 200)
xi_prob = (
    1.0
    / ((1 - xi) * np.sqrt(2 * np.pi * xi_nu_0))
    * np.exp(-((np.log(1 - xi) - xi_mu_0) ** 2) / (2 * xi_nu_0))
)
xi_source = ColumnDataSource(data=dict(xi=xi, xi_prob=xi_prob))
xi_plot = figure(
    plot_width=400,
    plot_height=400,
    sizing_mode="stretch_height",
    toolbar_location=None,
    x_range=(-1.01, 1.01),
    title="contrast distribution",
)
xi_plot.title.align = "center"
xi_plot.title.text_font_size = "14pt"
xi_plot.line("xi", "xi_prob", source=xi_source, line_width=3, line_alpha=0.6)
xi_plot.xaxis.axis_label = "spot contrast"
xi_plot.xaxis.axis_label_text_font_style = "normal"
xi_plot.xaxis.axis_label_text_font_size = "12pt"
xi_plot.yaxis[0].formatter = DisableTickLabels
xi_plot.yaxis.axis_label = "probability"
xi_plot.yaxis.axis_label_text_font_style = "normal"
xi_plot.yaxis.axis_label_text_font_size = "12pt"
xi_plot.outline_line_width = 1
xi_plot.outline_line_alpha = 1
xi_plot.outline_line_color = "black"
xi_slider_mu = Slider(
    start=-3,
    end=3,
    value=xi_mu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
)
xi_slider_nu = Slider(
    start=0.01,
    end=0.99,
    value=xi_nu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
    name="nu",
)
xi_layout = row(
    xi_plot,
    column(mu, xi_slider_mu, margin=(10, 10, 10, 10)),
    column(nu, xi_slider_nu, margin=(10, 10, 10, 10)),
)

# Latitude distribution
phi = np.linspace(-90, 90, 200)
alpha = phi_mu_0 * (1 / phi_nu_0 - 1)
beta = (1 - phi_mu_0) * (1 / phi_nu_0 - 1)
phi_prob = (
    gamma(alpha + beta)
    / (gamma(alpha) * gamma(beta))
    * 0.5
    * np.abs(np.sin(phi * np.pi / 180))
    * np.cos(phi * np.pi / 180) ** (alpha - 1)
    * (1 - np.cos(phi * np.pi / 180)) ** (beta - 1)
)
phi_source = ColumnDataSource(data=dict(phi=phi, phi_prob=phi_prob))
phi_plot = figure(
    plot_width=400,
    plot_height=400,
    sizing_mode="stretch_height",
    toolbar_location=None,
    x_range=(-95, 95),
    title="latitude distribution",
)
phi_plot.title.align = "center"
phi_plot.title.text_font_size = "14pt"
phi_plot.line("phi", "phi_prob", source=phi_source, line_width=3, line_alpha=0.6)
phi_plot.xaxis.axis_label = "spot latitude (degrees)"
phi_plot.xaxis.axis_label_text_font_style = "normal"
phi_plot.xaxis.axis_label_text_font_size = "12pt"
phi_plot.yaxis[0].formatter = DisableTickLabels
phi_plot.xaxis.ticker = FixedTicker(ticks=[-90, -60, -30, 0, 30, 60, 90])
phi_plot.yaxis.axis_label = "probability"
phi_plot.yaxis.axis_label_text_font_style = "normal"
phi_plot.yaxis.axis_label_text_font_size = "12pt"
phi_plot.outline_line_width = 1
phi_plot.outline_line_alpha = 1
phi_plot.outline_line_color = "black"
phi_slider_mu = Slider(
    start=0.01,
    end=0.99,
    value=phi_mu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
)
phi_slider_nu = Slider(
    start=0.01,
    end=0.99,
    value=phi_nu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
    name="nu",
)
phi_layout = row(
    phi_plot,
    column(mu, phi_slider_mu, margin=(10, 10, 10, 10)),
    column(nu, phi_slider_nu, margin=(10, 10, 10, 10)),
)

# Size distribution
r = np.linspace(0, 1, 200)
alpha = r_mu_0 * (1 / r_nu_0 - 1)
beta = (1 - r_mu_0) * (1 / r_nu_0 - 1)
r_prob = (
    gamma(alpha + beta)
    / (gamma(alpha) * gamma(beta))
    * r ** (alpha - 1)
    * (1 - r) ** (beta - 1)
)
r_source = ColumnDataSource(data=dict(r=r, r_prob=r_prob))
r_plot = figure(
    plot_width=400,
    plot_height=400,
    sizing_mode="stretch_height",
    toolbar_location=None,
    x_range=(-0.01, 1.01),
    title="radius distribution",
)
r_plot.title.align = "center"
r_plot.title.text_font_size = "14pt"
r_plot.line("r", "r_prob", source=r_source, line_width=3, line_alpha=0.6)
r_plot.xaxis.axis_label = "spot radius"
r_plot.xaxis.axis_label_text_font_style = "normal"
r_plot.xaxis.axis_label_text_font_size = "12pt"
r_plot.yaxis[0].formatter = DisableTickLabels
r_plot.yaxis.axis_label = "probability"
r_plot.yaxis.axis_label_text_font_style = "normal"
r_plot.yaxis.axis_label_text_font_size = "12pt"
r_plot.outline_line_width = 1
r_plot.outline_line_alpha = 1
r_plot.outline_line_color = "black"
r_slider_mu = Slider(
    start=0.01,
    end=0.99,
    value=r_mu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
)
r_slider_nu = Slider(
    start=0.01,
    end=0.99,
    value=r_nu_0,
    step=0.01,
    orientation="vertical",
    format="0.3f",
    css_classes=["custom-slider"],
    name="nu",
)
r_layout = row(
    r_plot,
    column(mu, r_slider_mu, margin=(10, 10, 10, 10)),
    column(nu, r_slider_nu, margin=(10, 10, 10, 10)),
)

# JS callback

callback = CustomJS(
    args=dict(
        r_source=r_source,
        r_slider_mu=r_slider_mu,
        r_slider_nu=r_slider_nu,
        phi_source=phi_source,
        phi_slider_mu=phi_slider_mu,
        phi_slider_nu=phi_slider_nu,
        xi_source=xi_source,
        xi_slider_mu=xi_slider_mu,
        xi_slider_nu=xi_slider_nu,
        ylm_source=ylm_source,
        A=A,
        ydeg=ydeg,
        ylm=ylm,
        mu_r=mu_r,
        nu_r=nu_r,
        mu_phi=mu_phi,
        nu_phi=nu_phi,
        mu_xi=mu_xi,
        nu_xi=nu_xi,
    ),
    code="""
        function gamma(num) {
            // https://www.w3resource.com/javascript-exercises/javascript-math-exercise-49.php
            var p = [
                0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059, 12.507343278686905, 
                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
            ];
            var i;
            var g = 7;
            if (num < 0.5) 
                return Math.PI / (Math.sin(Math.PI * num) * gamma(1 - num));
            num -= 1;
            var a = p[0];
            var t = num + g + 0.5;
            for (i = 1; i < p.length; i++) {
                a += p[i] / (num + i);
            }
            return Math.sqrt(2 * Math.PI) * Math.pow(t, num + 0.5) * Math.exp(-t) * a;
        }

        // Get the hyperparams
        var mu_r_user = r_slider_mu.value;
        var nu_r_user = r_slider_nu.value;
        var mu_phi_user = phi_slider_mu.value;
        var nu_phi_user = phi_slider_nu.value;
        var mu_xi_user = xi_slider_mu.value;
        var nu_xi_user = xi_slider_nu.value;

        // Get the closest point for which we have data
        var min_loss = 1e10;
        var best_k = 0;
        for (var k = 0; k < ylm.length; k++) {
            loss = (Math.pow(mu_r_user - mu_r[k], 2) + 
                    Math.pow(nu_r_user - nu_r[k], 2) + 
                    Math.pow(mu_phi_user - mu_phi[k], 2) + 
                    Math.pow(nu_phi_user - nu_phi[k], 2) + 
                    Math.pow(mu_xi_user - mu_xi[k], 2) + 
                    Math.pow(nu_xi_user - nu_xi[k], 2));
            if (loss < min_loss) {
                min_loss = loss;
                best_k = k;
            }
        }
        var k = best_k;

        // Force the sliders to the grid point
        r_slider_mu.value = mu_r[k];
        r_slider_nu.value = nu_r[k];
        phi_slider_mu.value = mu_phi[k];
        phi_slider_nu.value = nu_phi[k];
        xi_slider_mu.value = mu_xi[k];
        xi_slider_nu.value = nu_xi[k];

        // Update the sample plots
        var N = (ydeg + 1) * (ydeg + 1);
        for (var i = 0; i < ylm_source.length; i++) {

            // Compute the image
            for (var j = 0; j < ylm_source[i].data.image[0].length; j++) {
                var image_j = 0;
                for (var n = 0; n < N; n++) {
                    image_j += A[j][n] * ylm[k][i][n];
                }
                ylm_source[i].data.image[0][j] = image_j;
            }

            // Render it
            ylm_source[i].change.emit();

        }

        // Transformed quantities
        var alpha_r = mu_r[k] * (1 / nu_r[k] - 1);
        var beta_r = (1 - mu_r[k]) * (1 / nu_r[k] - 1);
        var alpha_phi = mu_phi[k] * (1 / nu_phi[k] - 1);
        var beta_phi = (1 - mu_phi[k]) * (1 / nu_phi[k] - 1);
        
        // Normalization constants
        var norm_r = gamma(alpha_r + beta_r) / (gamma(alpha_r) * gamma(beta_r));
        var norm_phi = gamma(alpha_phi + beta_phi) / (gamma(alpha_phi) * gamma(beta_phi));
        var norm_xi = 1.0 / (Math.sqrt(2 * Math.PI * nu_xi[k]));

        // Get the x, y arrays
        var x_r = r_source.data['r'];
        var y_r = r_source.data['r_prob'];
        var x_phi = phi_source.data['phi'];
        var y_phi = phi_source.data['phi_prob'];
        var x_xi = xi_source.data['xi'];
        var y_xi = xi_source.data['xi_prob'];

        // Compute the pdf
        for (var i = 0; i < x_r.length; i++) {
            var cos_phi = Math.cos(x_phi[i] * Math.PI / 180.0);
            var jac_phi = 0.5 * Math.abs(Math.sin(x_phi[i] * Math.PI / 180.0));
            y_r[i] = norm_r * Math.pow(x_r[i], alpha_r - 1) * Math.pow(1 - x_r[i], beta_r - 1);
            y_phi[i] = norm_phi * jac_phi * Math.pow(cos_phi, alpha_phi - 1) * Math.pow(1 - cos_phi, beta_phi - 1);
            y_xi[i] = norm_xi / (1 - x_xi[i]) * Math.exp(-Math.pow(Math.log(1 - x_xi[i]) - mu_xi[k], 2) / (2 * nu_xi[k]));
        }

        // Update
        r_source.change.emit();
        phi_source.change.emit();
        xi_source.change.emit();
    """,
    name="callback",
)
r_slider_mu.callback_policy = "mouseup"
r_slider_mu.js_on_change("value_throttled", callback)
r_slider_nu.callback_policy = "mouseup"
r_slider_nu.js_on_change("value_throttled", callback)
phi_slider_mu.callback_policy = "mouseup"
phi_slider_mu.js_on_change("value_throttled", callback)
phi_slider_nu.callback_policy = "mouseup"
phi_slider_nu.js_on_change("value_throttled", callback)
xi_slider_mu.callback_policy = "mouseup"
xi_slider_mu.js_on_change("value_throttled", callback)
xi_slider_nu.callback_policy = "mouseup"
xi_slider_nu.js_on_change("value_throttled", callback)

# Collect all distributions
distr_layout = row(phi_layout, r_layout, xi_layout, style_div)

# Collect and show everything
layout = column(distr_layout, ylm_layout)
show(layout)
