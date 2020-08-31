from bokeh.models import Div
from bokeh.palettes import Plasma256
import numpy as np
import os

__all__ = ["plasma", "svg_mu", "svg_nu", "loader", "style"]


# Plasma gradient
idx = np.array(np.linspace(0, 255, 101), dtype=int)
Plasma100 = np.array(Plasma256)[idx][::-1]
plasma = []
for v, hex_color in enumerate(Plasma100):
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    plasma.append("rgba({:d}, {:d}, {:d}, 1) {:d}%".format(r, g, b, v))
plasma = plasma[::3]
plasma = ",\n    ".join(plasma)
PLASMA = "background: linear-gradient(\n{:s}\n);".format(plasma)


# SVG: Greek mu
svg_mu = lambda: Div(
    text="""
<img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgoKPHN2ZwogICB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iCiAgIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiCiAgIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIKICAgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzb2RpcG9kaT0iaHR0cDovL3NvZGlwb2RpLnNvdXJjZWZvcmdlLm5ldC9EVEQvc29kaXBvZGktMC5kdGQiCiAgIHhtbG5zOmlua3NjYXBlPSJodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy9uYW1lc3BhY2VzL2lua3NjYXBlIgogICB3aWR0aD0iNDAwIgogICBoZWlnaHQ9IjQwMCIKICAgaWQ9InN2ZzQ2MTEiCiAgIHNvZGlwb2RpOnZlcnNpb249IjAuMzIiCiAgIGlua3NjYXBlOnZlcnNpb249IjAuOTIuNCAoNWRhNjg5YzMxMywgMjAxOS0wMS0xNCkiCiAgIHZlcnNpb249IjEuMCIKICAgc29kaXBvZGk6ZG9jbmFtZT0iR3JlZWtfbGNfbXUuc3ZnIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNDYxMyIgLz4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgaWQ9ImJhc2UiCiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEuMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMDAwMCIKICAgICBndWlkZXRvbGVyYW5jZT0iMTAiCiAgICAgb2JqZWN0dG9sZXJhbmNlPSIxLjYiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAuMCIKICAgICBpbmtzY2FwZTpwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnpvb209IjAuNyIKICAgICBpbmtzY2FwZTpjeD0iODUuOTA5MTkiCiAgICAgaW5rc2NhcGU6Y3k9IjE5MC42MTg4MyIKICAgICBpbmtzY2FwZTpkb2N1bWVudC11bml0cz0icHgiCiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0ibGF5ZXIxIgogICAgIGlua3NjYXBlOm9iamVjdC1iYm94PSJ0cnVlIgogICAgIGlua3NjYXBlOm9iamVjdC1wb2ludHM9InRydWUiCiAgICAgaW5rc2NhcGU6b2JqZWN0LW5vZGVzPSJ0cnVlIgogICAgIGlua3NjYXBlOmdyaWQtcG9pbnRzPSJ0cnVlIgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTkyMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctaGVpZ2h0PSIxMDE3IgogICAgIGlua3NjYXBlOndpbmRvdy14PSItOCIKICAgICBpbmtzY2FwZTp3aW5kb3cteT0iLTgiCiAgICAgd2lkdGg9IjQwMHB4IgogICAgIGhlaWdodD0iNDAwcHgiCiAgICAgc2hvd2dyaWQ9ImZhbHNlIgogICAgIGlua3NjYXBlOndpbmRvdy1tYXhpbWl6ZWQ9IjEiIC8+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhNDYxNiI+CiAgICA8cmRmOlJERj4KICAgICAgPGNjOldvcmsKICAgICAgICAgcmRmOmFib3V0PSIiPgogICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2Uvc3ZnK3htbDwvZGM6Zm9ybWF0PgogICAgICAgIDxkYzp0eXBlCiAgICAgICAgICAgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIgLz4KICAgICAgPC9jYzpXb3JrPgogICAgPC9yZGY6UkRGPgogIDwvbWV0YWRhdGE+CiAgPGcKICAgICBpbmtzY2FwZTpsYWJlbD0iTGF5ZXIgMSIKICAgICBpbmtzY2FwZTpncm91cG1vZGU9ImxheWVyIgogICAgIGlkPSJsYXllcjEiCiAgICAgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI1Ni42Nzk4LC01MzEuNzk2MykiPgogICAgPGcKICAgICAgIGFyaWEtbGFiZWw9Is68IgogICAgICAgc3R5bGU9ImZvbnQtc3R5bGU6bm9ybWFsO2ZvbnQtd2VpZ2h0Om5vcm1hbDtsaW5lLWhlaWdodDowJTtmb250LWZhbWlseTonVGltZXMgTmV3IFJvbWFuJzt0ZXh0LWFsaWduOnN0YXJ0O3RleHQtYW5jaG9yOnN0YXJ0O2ZpbGw6IzAwMDAwMDtmaWxsLW9wYWNpdHk6MTtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MXB4O3N0cm9rZS1saW5lY2FwOmJ1dHQ7c3Ryb2tlLWxpbmVqb2luOm1pdGVyO3N0cm9rZS1vcGFjaXR5OjEiCiAgICAgICBpZD0idGV4dDU1NDAiPgogICAgICA8cGF0aAogICAgICAgICBkPSJtIDQ4NS43MzExNCw4MDcuOTQ1OCBxIC0zNS4zNTE1NiwzNy44OTA2MyAtNjEuNTIzNDQsMzcuODkwNjMgLTE2Ljc5Njg3LDAgLTMxLjI1LC0xMi44OTA2MyB2IDEwLjM1MTU2IHEgMCwxNy45Njg3NSA1LjQ2ODc1LDM2LjEzMjgyIDYuMjUsMjAuMTE3MTggNi4yNSwyNi41NjI1IDAsOC45ODQzNyAtNS42NjQwNiwxNC44NDM3NSAtNS42NjQwNiw1Ljg1OTM3IC0xNC4wNjI1LDUuODU5MzcgLTguNTkzNzUsMCAtMTMuNjcxODcsLTYuODM1OTQgLTUuMDc4MTMsLTYuODM1OTMgLTUuMDc4MTMsLTE2LjQwNjI1IDAsLTcuMDMxMjUgNS4wNzgxMywtMjUuMzkwNjIgNS44NTkzNywtMjAuMzEyNSA1Ljg1OTM3LC0zOS40NTMxMyBWIDY2MS40NjE0MyBoIDMyLjIyNjU2IHYgMTE1LjgyMDMxIHEgMCwxNy41NzgxMiAyLjkyOTY5LDI1Ljc4MTI1IDMuMTI1LDguMjAzMTIgMTAuNzQyMTksMTMuNDc2NTYgNy44MTI1LDUuMDc4MTMgMTcuNTc4MTIsNS4wNzgxMyAxNy4zODI4MiwwIDQ1LjExNzE5LC0yMy44MjgxMyBWIDY2MS40NjE0MyBoIDMyLjQyMTg4IHYgMTM1Ljc0MjE4IHEgMCwxNy4xODc1IDMuNTE1NjIsMjMuODI4MTMgMy41MTU2Myw2LjQ0NTMxIDExLjkxNDA2LDYuNDQ1MzEgMTMuMjgxMjUsMCAxNy4xODc1LC0yNS45NzY1NiBoIDcuMDMxMjUgcSAtMy43MTA5Myw0NC4zMzU5NCAtMzcuNSw0NC4zMzU5NCAtMTQuNjQ4NDMsMCAtMjQuNDE0MDYsLTkuNzY1NjMgLTkuNTcwMzEsLTkuOTYwOTQgLTEwLjE1NjI1LC0yOC4xMjUgeiIKICAgICAgICAgc3R5bGU9ImZvbnQtc2l6ZTo0MDBweDtsaW5lLWhlaWdodDoxLjI1O3RleHQtYWxpZ246Y2VudGVyO3RleHQtYW5jaG9yOm1pZGRsZSIKICAgICAgICAgaWQ9InBhdGg4MTQiIC8+CiAgICA8L2c+CiAgPC9nPgo8L3N2Zz4K"
width=20, height=20></img>
""",
    css_classes=["custom-slider-title"],
)


# SVG: Greek sigma
svg_sigma = lambda: Div(
    text="""
<img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIKICAgeG1sbnM6Y2M9Imh0dHA6Ly9jcmVhdGl2ZWNvbW1vbnMub3JnL25zIyIKICAgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIgogICB4bWxuczpzdmc9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnNvZGlwb2RpPSJodHRwOi8vc29kaXBvZGkuc291cmNlZm9yZ2UubmV0L0RURC9zb2RpcG9kaS0wLmR0ZCIKICAgeG1sbnM6aW5rc2NhcGU9Imh0dHA6Ly93d3cuaW5rc2NhcGUub3JnL25hbWVzcGFjZXMvaW5rc2NhcGUiCiAgIHdpZHRoPSIxMjUwIgogICBoZWlnaHQ9IjI1MDAiCiAgIGlkPSJzdmcyIgogICBzb2RpcG9kaTp2ZXJzaW9uPSIwLjMyIgogICBpbmtzY2FwZTp2ZXJzaW9uPSIwLjQ2ZGV2K2RldmVsIgogICB2ZXJzaW9uPSIxLjAiCiAgIHNvZGlwb2RpOmRvY25hbWU9IlRpbWVzIE5ldyBSb21hbiBHcmVlayBzbWFsbCBsZXR0ZXIgc2lnbWEuc3ZnIgogICBpbmtzY2FwZTpvdXRwdXRfZXh0ZW5zaW9uPSJvcmcuaW5rc2NhcGUub3V0cHV0LnN2Zy5pbmtzY2FwZSI+CiAgPGRlZnMKICAgICBpZD0iZGVmczQiIC8+CiAgPHNvZGlwb2RpOm5hbWVkdmlldwogICAgIGlkPSJiYXNlIgogICAgIHBhZ2Vjb2xvcj0iI2ZmZmZmZiIKICAgICBib3JkZXJjb2xvcj0iIzY2NjY2NiIKICAgICBib3JkZXJvcGFjaXR5PSIxLjAiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAuMCIKICAgICBpbmtzY2FwZTpwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnpvb209IjAuMDg3NSIKICAgICBpbmtzY2FwZTpjeD0iLTUzNy43NzgyMyIKICAgICBpbmtzY2FwZTpjeT0iLTU0LjQwNjEzNCIKICAgICBpbmtzY2FwZTpkb2N1bWVudC11bml0cz0icHgiCiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0ibGF5ZXIxIgogICAgIHNob3dncmlkPSJmYWxzZSIKICAgICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iZmFsc2UiCiAgICAgaW5rc2NhcGU6d2luZG93LXdpZHRoPSIxMjgwIgogICAgIGlua3NjYXBlOndpbmRvdy1oZWlnaHQ9IjEwMDMiCiAgICAgaW5rc2NhcGU6d2luZG93LXg9IjAiCiAgICAgaW5rc2NhcGU6d2luZG93LXk9IjAiIC8+CiAgPG1ldGFkYXRhCiAgICAgaWQ9Im1ldGFkYXRhNyI+CiAgICA8cmRmOlJERj4KICAgICAgPGNjOldvcmsKICAgICAgICAgcmRmOmFib3V0PSIiPgogICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2Uvc3ZnK3htbDwvZGM6Zm9ybWF0PgogICAgICAgIDxkYzp0eXBlCiAgICAgICAgICAgcmRmOnJlc291cmNlPSJodHRwOi8vcHVybC5vcmcvZGMvZGNtaXR5cGUvU3RpbGxJbWFnZSIgLz4KICAgICAgPC9jYzpXb3JrPgogICAgPC9yZGY6UkRGPgogIDwvbWV0YWRhdGE+CiAgPGcKICAgICBpbmtzY2FwZTpsYWJlbD0iTGl2ZWxsbyAxIgogICAgIGlua3NjYXBlOmdyb3VwbW9kZT0ibGF5ZXIiCiAgICAgaWQ9ImxheWVyMSIKICAgICB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1MzIuMTQyODgsMTM1OS4wNjY0KSI+CiAgICA8cGF0aAogICAgICAgc3R5bGU9ImZvbnQtc2l6ZToyMDQ4cHg7Zm9udC1zdHlsZTpub3JtYWw7Zm9udC13ZWlnaHQ6bm9ybWFsO3RleHQtYWxpZ246Y2VudGVyO3RleHQtYW5jaG9yOm1pZGRsZTtmaWxsOiMwMDAwMDA7ZmlsbC1vcGFjaXR5OjE7ZmlsbC1ydWxlOm5vbnplcm87c3Ryb2tlOm5vbmU7c3Ryb2tlLXdpZHRoOjM7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46cm91bmQ7c3Ryb2tlLW1pdGVybGltaXQ6MjtzdHJva2UtZGFzaG9mZnNldDowO3N0cm9rZS1vcGFjaXR5OjE7Zm9udC1mYW1pbHk6VGltZXMgTmV3IFJvbWFuIgogICAgICAgZD0iTSA1ODMuODU3MTIsLTM0Ny41NjY0MSBMIDU4My44NTcxMiwtMjAxLjU2NjQxIEwgMTkzLjg1NzEyLC0yMDEuNTY2NDEgQyAyNzkuMTg5NjksLTE1Mi44OTkwMiAzNDguODU2MjksLTkxLjM5OTA4IDQwMi44NTcxMiwtMTcuMDY2NDA2IEMgNDU2Ljg1NjE4LDU3LjI2NzQzOCA0ODMuODU2MTUsMTM2LjEwMDY5IDQ4My44NTcxMiwyMTkuNDMzNTkgQyA0ODMuODU2MTUsMzI4LjEwMDUgNDQxLjAyMjg2LDQxOC4xMDA0MSAzNTUuMzU3MTIsNDg5LjQzMzU5IEMgMjY5LjY4OTcsNTYwLjc2NjkzIDE3Mi41MjMxMyw1OTYuNDMzNTcgNjMuODU3MTE3LDU5Ni40MzM1OSBDIC02NC44MDk5NjQsNTk2LjQzMzU3IC0xNzUuODA5ODUsNTQ1LjkzMzYyIC0yNjkuMTQyODgsNDQ0LjkzMzU5IEMgLTM2Mi40NzYzMywzNDMuOTMzODIgLTQwOS4xNDI5NSwyMjYuNDMzOTQgLTQwOS4xNDI4OCw5Mi40MzM1OTQgQyAtNDA5LjE0Mjk1LC0yLjIzMjUwMjMgLTM4NC42NDI5OCwtODUuODk5MDg1IC0zMzUuNjQyODgsLTE1OC41NjY0MSBDIC0yODYuNjQzMDgsLTIzMS4yMzIyNyAtMjI5LjY0MzEzLC0yODAuODk4ODkgLTE2NC42NDI4OCwtMzA3LjU2NjQxIEMgLTk5LjY0MzI2MywtMzM0LjIzMjE3IC0xMS44MTAwMTcsLTM0Ny41NjU0OSA5OC44NTcxMTcsLTM0Ny41NjY0MSBMIDU4My44NTcxMiwtMzQ3LjU2NjQxIHogTSAxMjYuODU3MTIsLTIwMS41NjY0MSBMIDkxLjg1NzExNywtMjAxLjU2NjQxIEMgLTguMTQzMzU0MywtMjAxLjU2NTY0IC04Ny45NzY2MDgsLTE3Ni43MzIzMyAtMTQ3LjY0Mjg4LC0xMjcuMDY2NDEgQyAtMjA3LjMwOTgyLC03Ny4zOTkwOTQgLTIzNy4xNDMxMywwLjQzNDE2MTc1IC0yMzcuMTQyODgsMTA2LjQzMzU5IEMgLTIzNy4xNDMxMywyMTkuMTAwNjEgLTIwNS44MDk4MiwzMTguNjAwNTEgLTE0My4xNDI4OCw0MDQuOTMzNTkgQyAtODAuNDc2NjE1LDQ5MS4yNjcgLTcuMTQzMzU1Myw1MzQuNDMzNjMgNzYuODU3MTE3LDUzNC40MzM1OSBDIDE0NC4xODk4Myw1MzQuNDMzNjMgMTk5Ljg1NjQ0LDUwMS45MzM2NiAyNDMuODU3MTIsNDM2LjkzMzU5IEMgMjg3Ljg1NjM1LDM3MS45MzM3OSAzMDkuODU2MzMsMjkzLjQzMzg3IDMwOS44NTcxMiwyMDEuNDMzNTkgQyAzMDkuODU2MzMsNTUuNDM0MTA3IDI0OC44NTYzOSwtNzguODk5MDkyIDEyNi44NTcxMiwtMjAxLjU2NjQxIEwgMTI2Ljg1NzEyLC0yMDEuNTY2NDEgeiIKICAgICAgIGlkPSJ0ZXh0MjMzMiIgLz4KICA8L2c+Cjwvc3ZnPgo="
width=10, height=20, style="margin-right:5px;"></img>
""",
    css_classes=["custom-slider-title"],
)


# Loading screen
loader = lambda: Div(
    text="""
<div class="preloader">
    <div class="spinner">
        <div class="dot1"></div>
        <div class="dot2"></div>
        <div class="loader-message">
            &nbsp;&nbsp;&nbsp;Loading...
            <div style="font-size: 8pt; font-weight: 100; width: 160px; margin-top: 10px;">
                This may take up to 15 seconds.
            </div>
        </div>
    </div>
</div>
"""
)


# Custom CSS
style = lambda: Div(
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
    .seed-button .bk-btn {
        padding: 6px 6px !important;
        transform: rotate(-90deg);
        background-color: #ffe0c6 !important;
        margin-top: 45px;
        height: 30px;
        margin-left: -30px;
    }
    .colorbar-slider .bk-noUi-draggable {
        %s
    }
    .preloader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%%;
        height: 100%%;
        z-index: 99999;
        display: flex;
        flex-flow: row nowrap;
        justify-content: center;
        align-items: center;
        background: none repeat scroll 0 0 #ffffff;
    }
    .loader-message {
        font-size: 20pt;
        font-weight: 600;
        color: #999;
        margin: 100px auto;
    }
    .spinner {
        margin: 100px auto;
        width: 80px;
        height: 80px;
        position: relative;
        text-align: center;
    }
    .dot1 {
        width: 100%%;
        height: 100%%;
        display: inline-block;
        position: absolute;
        top: 0;
        background-color: #999;
        border-radius: 100%%;
        z-index: 1;
    }
    .dot2 {
        display: inline-block;
        position: absolute;
        background-color: #666;
        border-radius: 100%%;
        width: 40%%;
        height: 40%%;
        top: 60px;
        left: 0px;
        z-index: 2;
        -webkit-animation: sk-orbit 0.75s infinite linear;
        animation: sk-orbit 0.75s infinite linear;
    }
    @keyframes sk-orbit {
        0%% { z-index: 2; transform: translate(0%%, 0%%); }
        49%% { z-index: 2; transform: translate(400%%, -200%%); }
        50%% { z-index: 0; transform: translate(400%%, -200%%); }
        99%% { z-index: 0; transform: translate(0%%, 0%%); }
        100%% { z-index: 2; transform: translate(0%%, 0%%); }
    }
</style>
"""
    % PLASMA
)