from .css import loader, style, TEMPLATE
from .design import get_intensity_design_matrix, get_flux_design_matrix
from .moll import get_latitude_lines, get_longitude_lines
from starry_process import StarryProcess
import numpy as np
from numpy.linalg import LinAlgError
import theano
import theano.tensor as tt
import sys
import os
import time
import bokeh
from bokeh.layouts import column, row, grid
from bokeh.models import (
    ColumnDataSource,
    Slider,
    ColorBar,
    Label,
    RangeSlider,
    CustomJS,
    Button,
    Span,
    HoverTool,
)
from bokeh.plotting import figure, curdoc
from bokeh.models.tickers import FixedTicker
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import OrRd6
from bokeh.server.server import Server


# Parameter ranges & default values
params = {
    "latitude": {
        "mu": {"start": 0.0, "stop": 90.0, "step": 0.01, "value": 30.0},
        "sigma": {"start": 1.0, "stop": 30.0, "step": 0.01, "value": 5.0},
    },
    "size": {
        "mu": {"start": 0.0, "stop": 90.0, "step": 0.1, "value": 20.0},
        "sigma": {"start": 1.0, "stop": 30.0, "step": 0.1, "value": 5.0},
    },
    "contrast": {
        "mu": {"start": -0.99, "stop": 0.99, "step": 0.01, "value": 0.80},
        "sigma": {"start": 0.01, "stop": 1.0, "step": 0.01, "value": 0.05},
    },
}


def fluxnorm(x, **kwargs):
    return 1e3 * ((1 + x) / np.median(1 + x, **kwargs) - 1)


class Samples(object):
    def __init__(self, ydeg, npix, npts, debug=False):
        # Settings
        self.npix = npix
        self.npts = npts
        self.nmaps = 5

        # Design matrices
        self.A_I = get_intensity_design_matrix(ydeg, npix)
        self.A_F = get_flux_design_matrix(ydeg, npts)

        # Compile the GP
        sa = tt.dscalar()
        sb = tt.dscalar()
        la = tt.dscalar()
        lb = tt.dscalar()
        ca = tt.dscalar()
        cb = tt.dscalar()
        self.gp = StarryProcess(
            ydeg, sa=sa, sb=sb, la=la, lb=lb, ca=ca, cb=cb,
        )
        self.gp.random.seed(238)

        print("Compiling...")
        if debug:
            self.sample_ylm = lambda *args: [
                np.random.randn(self.nmaps, (ydeg + 1) ** 2)
            ]
        else:
            function = theano.function(
                [sa, sb, la, lb, ca, cb,],
                [self.gp.sample_ylm(self.nmaps)],
                no_default_updates=True,
            )

            def sample_ylm(mu_s, sigma_s, mu_l, sigma_l, mu_c, sigma_c):
                sa, sb = self.gp.size.transform.transform(mu_s, sigma_s)
                la, lb = self.gp.latitude.transform.transform(mu_l, sigma_l)
                ca = mu_c
                cb = sigma_c
                return function(sa, sb, la, lb, ca, cb)

            self.sample_ylm = sample_ylm

        print("Done!")

        # Draw three samples from the default distr
        self.ylm = self.sample_ylm(
            params["size"]["mu"]["value"],
            params["size"]["sigma"]["value"],
            params["latitude"]["mu"]["value"],
            params["latitude"]["sigma"]["value"],
            params["contrast"]["mu"]["value"],
            params["contrast"]["sigma"]["value"],
        )[0]

        # Plot the GP ylm samples
        self.color_mapper = LinearColorMapper(
            palette="Plasma256", nan_color="white", low=0.5, high=1.2
        )
        self.moll_plot = [None for i in range(self.nmaps)]
        self.moll_source = [
            ColumnDataSource(
                data=dict(
                    image=[
                        1.0
                        + (self.A_I @ self.ylm[i]).reshape(
                            self.npix, 2 * self.npix
                        )
                    ]
                )
            )
            for i in range(self.nmaps)
        ]
        eps = 0.1
        epsp = 0.02
        xe = np.linspace(-2, 2, 300)
        ye = 0.5 * np.sqrt(4 - xe ** 2)
        for i in range(self.nmaps):
            self.moll_plot[i] = figure(
                plot_width=280,
                plot_height=130,
                toolbar_location=None,
                x_range=(-2 - eps, 2 + eps),
                y_range=(-1 - eps / 2, 1 + eps / 2),
            )
            self.moll_plot[i].axis.visible = False
            self.moll_plot[i].grid.visible = False
            self.moll_plot[i].outline_line_color = None
            self.moll_plot[i].image(
                image="image",
                x=-2,
                y=-1,
                dw=4 + epsp,
                dh=2 + epsp / 2,
                color_mapper=self.color_mapper,
                source=self.moll_source[i],
            )
            self.moll_plot[i].toolbar.active_drag = None
            self.moll_plot[i].toolbar.active_scroll = None
            self.moll_plot[i].toolbar.active_tap = None

        # Plot lat/lon grid
        lat_lines = get_latitude_lines()
        lon_lines = get_longitude_lines()
        for i in range(self.nmaps):
            for x, y in lat_lines:
                self.moll_plot[i].line(
                    x, y, line_width=1, color="black", alpha=0.25
                )
            for x, y in lon_lines:
                self.moll_plot[i].line(
                    x, y, line_width=1, color="black", alpha=0.25
                )
            self.moll_plot[i].line(
                xe, ye, line_width=3, color="black", alpha=1
            )
            self.moll_plot[i].line(
                xe, -ye, line_width=3, color="black", alpha=1
            )

        # Colorbar slider
        self.slider = RangeSlider(
            start=0,
            end=1.5,
            step=0.01,
            value=(0.5, 1.2),
            orientation="horizontal",
            show_value=False,
            css_classes=["colorbar-slider"],
            direction="ltr",
            title="intensity range",
        )
        self.slider.on_change("value", self.slider_callback)

        # Seed button
        self.seed_button = Button(
            label="new randomizer seed",
            button_type="default",
            css_classes=["seed-button"],
            sizing_mode="fixed",
            width=142,
        )
        self.seed_button.on_click(self.seed_callback)

        self.continuous_button = Button(
            label="continuous update",
            button_type="default",
            css_classes=["continuous-button"],
            sizing_mode="fixed",
            width=142,
        )
        self.continuous_button.on_click(self.continuous_callback)

        # Light curve samples
        self.flux_plot = [None for i in range(self.nmaps)]
        self.flux_source = [
            ColumnDataSource(
                data=dict(
                    xs=[np.linspace(0, 2, npts) for j in range(6)],
                    ys=[fluxnorm(self.A_F[j] @ self.ylm[i]) for j in range(6)],
                    color=[OrRd6[5 - j] for j in range(6)],
                    inc=[15, 30, 45, 60, 75, 90],
                )
            )
            for i in range(self.nmaps)
        ]
        for i in range(self.nmaps):
            self.flux_plot[i] = figure(
                toolbar_location=None,
                x_range=(0, 2),
                y_range=None,
                min_border_left=50,
            )
            if i == 0:
                self.flux_plot[i].yaxis.axis_label = "flux [ppt]"
                self.flux_plot[i].yaxis.axis_label_text_font_style = "normal"
            self.flux_plot[i].xaxis.axis_label = "rotational phase"
            self.flux_plot[i].xaxis.axis_label_text_font_style = "normal"
            self.flux_plot[i].outline_line_color = None
            self.flux_plot[i].multi_line(
                xs="xs",
                ys="ys",
                line_color="color",
                source=self.flux_source[i],
            )
            self.flux_plot[i].toolbar.active_drag = None
            self.flux_plot[i].toolbar.active_scroll = None
            self.flux_plot[i].toolbar.active_tap = None
            self.flux_plot[i].yaxis.major_label_orientation = np.pi / 4
            self.flux_plot[i].xaxis.axis_label_text_font_size = "8pt"
            self.flux_plot[i].xaxis.major_label_text_font_size = "8pt"
            self.flux_plot[i].yaxis.axis_label_text_font_size = "8pt"
            self.flux_plot[i].yaxis.major_label_text_font_size = "8pt"

        # Legend
        for j, label in enumerate(["15", "30", "45", "60", "75", "90"]):
            self.flux_plot[-1].line(
                [0, 0], [0, 0], legend_label=label, line_color=OrRd6[5 - j]
            )
        self.flux_plot[-1].legend.location = "bottom_right"
        self.flux_plot[-1].legend.title = "inclination"
        self.flux_plot[-1].legend.title_text_font_style = "bold"
        self.flux_plot[-1].legend.title_text_font_size = "8pt"
        self.flux_plot[-1].legend.label_text_font_size = "8pt"
        self.flux_plot[-1].legend.spacing = 0
        self.flux_plot[-1].legend.label_height = 5
        self.flux_plot[-1].legend.glyph_height = 5

        # Full layout
        self.plots = row(
            *[
                column(m, f, sizing_mode="scale_both")
                for m, f in zip(self.moll_plot, self.flux_plot)
            ],
            margin=(10, 30, 10, 30),
            sizing_mode="scale_both",
            css_classes=["samples"],
        )
        self.layout = grid(
            [
                [self.slider],
                [self.plots],
                [self.continuous_button, self.seed_button],
            ]
        )

    def slider_callback(self, attr, old, new):
        self.color_mapper.low, self.color_mapper.high = self.slider.value

    def seed_callback(self, event):
        self.gp.random.seed(np.random.randint(0, 999))
        self.callback(None, None, None)

    def continuous_callback(self, event):
        if self.continuous_button.label == "continuous update":
            self.continuous_button.label = "discrete update"
            self.Latitude.throttle_time = 0.20
            self.Size.throttle_time = 0.20
            self.Contrast.throttle_time = 0.20
        else:
            self.continuous_button.label = "continuous update"
            self.Latitude.throttle_time = 0
            self.Size.throttle_time = 0
            self.Contrast.throttle_time = 0

    def callback(self, attr, old, new):
        try:

            # Draw the samples
            self.ylm = self.sample_ylm(
                self.Size.slider_mu.value,
                self.Size.slider_sigma.value,
                self.Latitude.slider_mu.value,
                self.Latitude.slider_sigma.value,
                self.Contrast.slider_mu.value,
                self.Contrast.slider_sigma.value,
            )[0]

            # Compute the images & plot the light curves
            for i in range(len(self.moll_source)):
                self.moll_source[i].data["image"] = [
                    1.0
                    + (self.A_I @ self.ylm[i]).reshape(
                        self.npix, 2 * self.npix
                    )
                ]

                self.flux_source[i].data["ys"] = [
                    fluxnorm(self.A_F[j] @ self.ylm[i])
                    for j in range(len(self.A_F))
                ]

            for dist in [self.Size, self.Latitude, self.Contrast]:
                for slider in [dist.slider_mu, dist.slider_sigma]:
                    slider.bar_color = "white"

        except Exception as e:

            # Something went wrong inverting the covariance!
            for dist in [self.Size, self.Latitude, self.Contrast]:
                for slider in [dist.slider_mu, dist.slider_sigma]:
                    slider.bar_color = "firebrick"

            print("An error occurred when computing the covariance:")
            print(e)


class Distribution(object):
    def __init__(self, name, xmin, xmax, mu, sigma, pdf, gp_callback):
        # Store
        self.pdf = pdf
        self.gp_callback = gp_callback
        self.last_run = 0.0
        self.throttle_time = 0.0

        # Arrays
        x = np.linspace(xmin, xmax, 300)
        y = self.pdf(x, mu["value"], sigma["value"])
        self.source = ColumnDataSource(data=dict(x=x, y=y))

        # Plot them
        dx = (xmax - xmin) * 0.01
        self.plot = figure(
            plot_width=400,
            plot_height=600,
            toolbar_location=None,
            x_range=(xmin - dx, xmax + dx),
            title="{} distribution".format(name),
            sizing_mode="stretch_both",
        )
        self.plot.title.align = "center"
        self.plot.title.text_font_size = "14pt"
        self.plot.line(
            "x", "y", source=self.source, line_width=3, line_alpha=0.6
        )
        # self.plot.xaxis.axis_label = name
        self.plot.xaxis.axis_label_text_font_style = "normal"
        self.plot.xaxis.axis_label_text_font_size = "12pt"
        self.plot.yaxis[0].formatter = FuncTickFormatter(code="return '  ';")
        self.plot.yaxis.axis_label = "probability"
        self.plot.yaxis.axis_label_text_font_style = "normal"
        self.plot.yaxis.axis_label_text_font_size = "12pt"
        self.plot.outline_line_width = 1
        self.plot.outline_line_alpha = 1
        self.plot.outline_line_color = "black"
        self.plot.toolbar.active_drag = None
        self.plot.toolbar.active_scroll = None
        self.plot.toolbar.active_tap = None

        # Sliders
        self.slider_mu = Slider(
            start=mu["start"],
            end=mu["stop"],
            step=mu["step"],
            value=mu["value"],
            orientation="horizontal",
            format="0.3f",
            css_classes=["custom-slider"],
            sizing_mode="stretch_width",
            height=10,
            show_value=False,
            title="μ",
        )
        self.slider_mu.on_change("value_throttled", self.callback)
        self.slider_mu.on_change("value", self.callback_throttled)
        self.slider_sigma = Slider(
            start=sigma["start"],
            end=sigma["stop"],
            step=sigma["step"],
            value=sigma["value"],
            orientation="horizontal",
            format="0.3f",
            css_classes=["custom-slider"],
            name="sigma",
            sizing_mode="stretch_width",
            height=10,
            show_value=False,
            title="σ",
        )
        self.slider_sigma.on_change("value_throttled", self.callback)
        self.slider_sigma.on_change("value", self.callback_throttled)

        # Show mean and std. dev.
        self.mean_vline = Span(
            location=self.slider_mu.value,
            dimension="height",
            line_color="black",
            line_width=1,
            line_dash="dashed",
        )
        self.std_vline1 = Span(
            location=self.slider_mu.value - self.slider_sigma.value,
            dimension="height",
            line_color="black",
            line_width=1,
            line_dash="dotted",
        )
        self.std_vline2 = Span(
            location=self.slider_mu.value + self.slider_sigma.value,
            dimension="height",
            line_color="black",
            line_width=1,
            line_dash="dotted",
        )
        self.plot.renderers.extend(
            [self.mean_vline, self.std_vline1, self.std_vline2]
        )

        # Full layout
        self.layout = grid(
            [
                [self.plot],
                [
                    column(
                        self.slider_mu,
                        self.slider_sigma,
                        sizing_mode="stretch_width",
                    )
                ],
            ]
        )

    def callback(self, attr, old, new):
        try:

            # Update the distribution
            self.source.data["y"] = self.pdf(
                self.source.data["x"],
                self.slider_mu.value,
                self.slider_sigma.value,
            )
            self.slider_mu.bar_color = "white"
            self.slider_sigma.bar_color = "white"
            self.plot.background_fill_color = "white"
            self.plot.background_fill_alpha = 1

            # Update the mean and std. dev. lines
            self.mean_vline.location = self.slider_mu.value
            self.std_vline1.location = (
                self.slider_mu.value - self.slider_sigma.value
            )
            self.std_vline2.location = (
                self.slider_mu.value + self.slider_sigma.value
            )

        except Exception as e:

            # A param is out of bounds!
            if "std" in str(e) or "sigma" in str(e):
                self.slider_sigma.bar_color = "firebrick"
            elif "mean" in str(e) or "mu" in str(e):
                self.slider_mu.bar_color = "firebrick"
            else:
                print("An error occurred when setting the parameters:")
                print(e)
            self.plot.background_fill_color = "firebrick"
            self.plot.background_fill_alpha = 0.2

        else:

            # Update the GP samples
            self.gp_callback(attr, old, new)

    def callback_throttled(self, attr, old, new):
        # manual throttling (not perfect)
        if self.throttle_time > 0:
            now = time.time()
            if now - self.last_run >= self.throttle_time:
                self.last_run = now
                return self.callback(attr, old, new)


class Application(object):
    def __init__(self, doc, ydeg=15, npix=100, npts=300, debug=False):

        self.ydeg = ydeg
        self.npix = npix
        self.npts = npts
        self.debug = debug

        # Display the loader
        self.layout = column(loader(), style(), sizing_mode="scale_both")
        doc.add_root(self.layout)
        doc.title = "starry process"
        doc.template = TEMPLATE

        # Set up the starry process
        doc.add_timeout_callback(self.run, 1000)

    def run(self):

        # The GP samples
        self.Samples = Samples(
            self.ydeg, self.npix, self.npts, debug=self.debug
        )

        # The distributions
        self.Latitude = Distribution(
            "latitude",
            -90,
            90,
            params["latitude"]["mu"],
            params["latitude"]["sigma"],
            lambda x, mu, sigma: self.Samples.gp.latitude.transform.pdf(
                x, mu=mu, sigma=sigma
            ),
            self.Samples.callback,
        )
        self.Size = Distribution(
            "size",
            0,
            90,
            params["size"]["mu"],
            params["size"]["sigma"],
            lambda x, mu, sigma: self.Samples.gp.size.transform.pdf(
                x, mu=mu, sigma=sigma
            ),
            self.Samples.callback,
        )
        self.Contrast = Distribution(
            "contrast",
            -1,
            1,
            params["contrast"]["mu"],
            params["contrast"]["sigma"],
            self.Samples.gp.contrast.transform.pdf,
            self.Samples.callback,
        )

        # Tell the GP about the sliders
        self.Samples.Latitude = self.Latitude
        self.Samples.Size = self.Size
        self.Samples.Contrast = self.Contrast

        # Full layout
        self.layout.children.pop(0)
        self.layout.children.append(
            column(
                row(
                    self.Latitude.layout,
                    self.Size.layout,
                    self.Contrast.layout,
                    sizing_mode="scale_both",
                ),
                self.Samples.layout,
                sizing_mode="scale_both",
            )
        )


def main():
    server = Server({"/": Application})
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
