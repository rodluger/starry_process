from bokeh.models import Div
from bokeh.palettes import Plasma256
import numpy as np
import os

__all__ = ["plasma", "svg_mu", "svg_nu", "loader", "style"]

# jinja template
TEMPLATE = """
{% from macros import embed %}
<!DOCTYPE html>
<html>
  {% block head %}
  <head>
    {% block inner_head %}
      <meta charset="utf-8">
      <title>{% block title %}{{ title | e if title else "Bokeh Plot" }}{% endblock %}</title>
      {% block preamble %}{% endblock %}
      {% block resources %}
        {% block css_resources %}
          {{ bokeh_css | indent(8) if bokeh_css }}
        {% endblock %}
        {% block js_resources %}
          {{ bokeh_js | indent(8) if bokeh_js }}
        {% endblock %}
      {% endblock %}
      {% block postamble %}{% endblock %}
    {% endblock %}
  </head>
  {% endblock %}
  {% block body %}
  <body style="min-width: 700px; !important">
    <div style="width:90%; height:90%; margin-left: auto; margin-right: auto;">
    {% block inner_body %}
      {% block contents %}
        {% for doc in docs %}
          {{ embed(doc) if doc.elementid }}
          {% for root in doc.roots %}
            {% block root scoped %}
              {{ embed(root) | indent(10) }}
            {% endblock %}
          {% endfor %}
        {% endfor %}
      {% endblock %}
      {{ plot_script | indent(8) }}
    {% endblock %}
    </div>
  </body>
  {% endblock %}
</html>
"""

# Plasma gradient
idx = np.array(np.linspace(0, 255, 101), dtype=int)
Plasma100 = np.array(Plasma256)[idx][::-1]
plasma = []
for v, hex_color in enumerate(Plasma100):
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    plasma.append("rgba({:d}, {:d}, {:d}, 1) {:d}%".format(r, g, b, v))
plasma = plasma[::3]
plasma = ",\n    ".join(plasma)
PLASMA = "background: linear-gradient(to right, \n{:s}\n);".format(plasma)


# Loading screen
loader = lambda: Div(
    text="""
<div class="preloader">
    <div class="spinner">
        <div class="dot1"></div>
        <div class="dot2"></div>
        <div class="loader-message">
            &nbsp;Compiling...
            <div style="font-size: 8pt; font-weight: 100; width: 160px; margin-top: 10px;">
                This may take up to 60 orbits.
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
    html, body {
        margin: 0 !important; 
        height: 100%% !important; 
        overflow: hidden !important;
    }
    .samples{
        margin-top: 50px !important;
    }
    .custom-slider .bk-input-group {
        padding-left: 30px !important;
        padding-right: 0px !important;
        margin-left: 30px !important;
        margin-right: 0px !important;
    }
    .custom-slider .bk-slider-title {
        position: absolute;
        left: 35px;
        font-weight: 600;
    }
    .colorbar-slider {
        margin-top: 25px !important;
        padding-right: 180px !important;
    }
    .colorbar-slider .bk-input-group {
        padding-left: 130px !important;
        padding-right: 130px !important;
        margin-left: 30px !important;
        margin-right: 0px !important;
    }
    .colorbar-slider .bk-noUi-draggable {
        %s
    }
    .colorbar-slider .bk-slider-title {
        position: absolute;
        left: 35px;
        font-weight: 600;
    }

    .seed-button {
        left: auto !important;
        right: 0 !important;
        position: absolute !important;
        margin-right: 0px !important;
        top: 25px !important;
    }

    .continuous-button {
        left: auto !important;
        right: 100px !important;
        position: absolute !important;
        margin-right: 50px !important;
        top: 25px !important;
    }

    /* Loading screen */

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
        position: absolute;
        left: -40px;
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
        right: 0px;
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
        left: -40px;
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
