from bokeh.models import Div
from bokeh.palettes import Plasma256, Plasma6
import numpy as np
import os

__all__ = ["plasma", "svg_mu", "svg_nu", "description", "style"]

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
    <style>
    .body-wrapper {
        width:90%;
        height:90%;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
  </head>
  {% endblock %}
  {% block body %}
  <body style="min-width: 1400px; !important">
    <div class="body-wrapper">
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

LOADING_SCREEN = """
<style>
    .preloader {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 99999;
        display: flex;
        flex-flow: row nowrap;
        justify-content: center;
        align-items: center;
        background: none repeat scroll 0 0 #ffffff;
    }
    .loader-message {
        font-family: Helvetica, Arial, sans-serif;
        font-size: 20pt;
        font-weight: 600;
        color: #999;
        margin: 100px auto;
        position: relative;
    }
    .spinner {
        margin: 100px auto;
        width: 320px;
        height: 320px;
        position: relative;
        text-align: center;
    }
    .dot1 {
        width: 25%;
        height: 25%;
        display: inline-block;
        position: absolute;
        left: 120px;
        top: 0;
        background-color: #999;
        border-radius: 100%;
        z-index: 1;
    }
    .dot2 {
        display: inline-block;
        position: absolute;
        background-color: #666;
        border-radius: 100%;
        width: 10%;
        height: 10%;
        top: 60px;
        left: 80px;
        z-index: 2;
        -webkit-animation: sk-orbit 0.75s infinite linear;
        animation: sk-orbit 0.75s infinite linear;
    }
    @keyframes sk-orbit {
        0% { z-index: 2; transform: translate(0%, 0%); }
        49% { z-index: 2; transform: translate(400%, -200%); }
        50% { z-index: 0; transform: translate(400%, -200%); }
        99% { z-index: 0; transform: translate(0%, 0%); }
        100% { z-index: 2; transform: translate(0%, 0%); }
    }
</style>
<div class="preloader">
    <div class="spinner">
        <div class="dot1"></div>
        <div class="dot2"></div>
        <div class="loader-message">
            &nbsp;starry process
            <div style="font-size: 10pt; font-weight: 100; margin-top: 10px;">
                sit tight while the page renders...
            </div>
        </div>
    </div>
</div>
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
PLASMA = "background: linear-gradient(to left, \n{:s}\n);".format(plasma)


# Description textbox
description = """
The sliders to the left and at the top control the hyperparameters of a
<a href="https://github.com/rodluger/starry_process"
style="text-decoration: none; font-weight:600; color: #444444;">
starry process</a>,
an interpretable gaussian process for stellar light curves.
The hyperparameters describe the spot latitude distribution
(<span style="font-style:italic;=">left</span>),
the spot radius
(<span style="font-style:italic;=">center</span>),
and the spot contrast
(<span style="font-style:italic;=">above</span>).
Below are five
samples from the process seen in a Mollweide projection on the stellar
surface, followed by the corresponding light curves viewed at
inclinations of
<span style="font-weight:600; color:{};">15</span>,
<span style="font-weight:600; color:{};">30</span>,
<span style="font-weight:600; color:{};">45</span>,
<span style="font-weight:600; color:{};">60</span>,
<span style="font-weight:600; color:{};">75</span>,
and
<span style="font-weight:600; color:{};">90</span>
degrees.
""".format(
    *[Plasma6[5 - j] for j in range(6)]
)


# Custom CSS
style = lambda: Div(
    text="""
<style>
    html, body {
        margin: 0 !important;
        height: 100%% !important;
    }
    .samples{
        margin-top: 10px !important;
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
    .bk.colorbar-slider {
        top: 65px !important;
    }
    .colorbar-slider {
        margin-top: 35px !important;
    }
    .colorbar-slider .bk-input-group {
        padding-left: 60px !important;
        padding-right: 0px !important;
        margin-left: 30px !important;
        margin-right: 0px !important;
    }
    .colorbar-slider .noUi-draggable {
        %s
    }
    .colorbar-slider .bk-slider-title {
        position: absolute;
        left: 35px;
        font-weight: 600;
    }
    .hidden-slider {
        visibility: hidden;
    }

    .smooth-button {
        position: relative !important;
        margin-top: 30px !important;
        display: inline-block !important;
        left: 20px !important;
    }

    .auto-button {
        position: relative !important;
        margin-top: 30px !important;
        display: inline-block !important;
        left: 32px !important;
    }

    .seed-button {
        position: relative !important;
        margin-top: 30px !important;
        display: inline-block !important;
        left: 44px !important;
    }

    .reset-button {
        position: relative !important;
        margin-top: 30px !important;
        display: inline-block !important;
        left: 56px !important;
    }

    .bk.button-row {
        position: relative !important;
        margin-left: auto !important;
        margin-right: auto !important;
        left: 0px !important;
        top: 60px !important;
    }

    .bk .control-title {
        position: relative !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .bk .control-title h1 {
        margin-left: 15px;
        font-size: 14pt;
        color: #444444;
        margin-top: 5px;
    }

    .bk .control-description {
        position: relative !important;
        margin-left: auto !important;
        margin-right: auto !important;
        top: 110px !important;
        padding-left: 35px !important;
        text-align: justify;
        color: #444444;
        left: 0px !important;
    }

    .bk .bk-slider-title {
        top: 2px !important;
    }

</style>
"""
    % PLASMA
)
