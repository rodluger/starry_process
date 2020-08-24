from css import svg_mu, svg_sigma, style
from design import get_design_matrix
from moll import get_latitude_lines, get_longitude_lines
import numpy as np
from numpy.linalg import LinAlgError
from starry_gp import YlmGP
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    Slider,
    ColorBar,
    Label,
    RangeSlider,
    CustomJS,
    Button,
    Span,
)
from bokeh.plotting import figure, curdoc
from bokeh.models.tickers import FixedTicker
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper

# Parameter ranges & default values
params = {
    "latitude": {
        "mu": {"start": 0.01, "stop": 0.99, "step": 0.01, "value": 0.8},
        "sigma": {"start": 0.01, "stop": 1.0, "step": 0.01, "value": 0.05},
    },
    "size": {
        "mu": {"start": 0.0, "stop": 90.0, "step": 0.1, "value": 15.0},
        "sigma": {"start": 1.0, "stop": 30.0, "step": 0.1, "value": 5.0},
    },
    "contrast": {
        "mu": {"start": -0.99, "stop": 0.99, "step": 0.01, "value": 0.80},
        "sigma": {"start": 0.01, "stop": 1.0, "step": 0.01, "value": 0.05},
    },
}


class Samples(object):
    def __init__(self, ydeg, npix, throttle):
        # Settings
        self.npix = npix
        self.nmaps = 5
        self.seed = 0

        # Intensity design matrix
        self.A = get_design_matrix(ydeg, npix)

        # The GP
        self.gp = YlmGP(ydeg)

        # Draw three samples from the default distr
        self.gp.size.set_params(
            params["size"]["mu"]["value"], params["size"]["sigma"]["value"]
        )
        self.gp.latitude.set_params(
            params["latitude"]["mu"]["value"],
            params["latitude"]["sigma"]["value"],
        )
        self.gp.contrast.set_params(
            params["contrast"]["mu"]["value"],
            params["contrast"]["sigma"]["value"],
        )
        np.random.seed(self.seed)
        self.ylm = self.gp.draw(self.nmaps)

        # Plot the GP ylm samples
        self.color_mapper = LinearColorMapper(
            palette="Plasma256", nan_color="white", low=0.5, high=1.2
        )
        self.plot = [None for i in range(self.nmaps)]
        self.source = [
            ColumnDataSource(
                data=dict(
                    image=[
                        1.0
                        + (self.A @ self.ylm[i]).reshape(
                            self.npix, 2 * self.npix
                        )
                    ]
                )
            )
            for i in range(self.nmaps)
        ]
        eps = 0.1
        for i in range(self.nmaps):
            self.plot[i] = figure(
                plot_width=280,
                plot_height=140,
                toolbar_location=None,
                x_range=(-2 - eps, 2 + eps),
                y_range=(-1 - eps / 2, 1 + eps / 2),
            )
            self.plot[i].axis.visible = False
            self.plot[i].grid.visible = False
            self.plot[i].outline_line_color = None
            self.plot[i].image(
                image="image",
                x=-2,
                y=-1,
                dw=4,
                dh=2,
                color_mapper=self.color_mapper,
                source=self.source[i],
            )

        # Plot lat/lon grid
        lat_lines = get_latitude_lines()
        lon_lines = get_longitude_lines()
        for i in range(self.nmaps):
            for x, y in lat_lines:
                self.plot[i].line(
                    x, y, line_width=1, color="black", alpha=0.25
                )
            for x, y in lon_lines:
                self.plot[i].line(
                    x, y, line_width=1, color="black", alpha=0.25
                )

            x = np.linspace(-2, 2, 300)
            y = 0.5 * np.sqrt(4 - x ** 2)
            self.plot[i].line(x, y, line_width=3, color="black", alpha=1)
            self.plot[i].line(x, -y, line_width=3, color="black", alpha=1)

        # Colorbar slider
        self.slider = RangeSlider(
            start=0,
            end=1.5,
            step=0.01,
            value=(0.5, 1.2),
            orientation="vertical",
            show_value=False,
            css_classes=["colorbar-slider"],
            direction="rtl",
            callback_policy="throttle",
            callback_throttle=throttle,
            height=118,
        )
        self.slider.on_change("value_throttled", self.slider_callback)

        # Seed button
        self.button = Button(
            label="re-seed",
            button_type="default",
            width=10,
            css_classes=["seed-button"],
        )
        self.button.on_click(self.seed_callback)

        # Full layout
        self.layout = row(
            *self.plot, self.slider, self.button, margin=(10, 30, 10, 30)
        )

    def slider_callback(self, attr, old, new):
        self.color_mapper.low, self.color_mapper.high = self.slider.value

    def seed_callback(self, event):
        self.seed = np.random.randint(0, 999)
        self.callback(None, None, None)

    def callback(self, attr, old, new):
        try:

            # Draw the samples
            self.gp.size.set_params(
                self.Size.slider_mu.value, self.Size.slider_sigma.value
            )
            self.gp.latitude.set_params(
                self.Latitude.slider_mu.value, self.Latitude.slider_sigma.value
            )
            self.gp.contrast.set_params(
                self.Contrast.slider_mu.value, self.Contrast.slider_sigma.value
            )
            np.random.seed(self.seed)
            self.ylm = self.gp.draw(self.nmaps)

            # Compute the images
            for i in range(len(self.source)):
                self.source[i].data["image"] = [
                    1.0
                    + (self.A @ self.ylm[i]).reshape(self.npix, 2 * self.npix)
                ]

            for dist in [self.Size, self.Latitude, self.Contrast]:
                for slider in [dist.slider_mu, dist.slider_sigma]:
                    slider.bar_color = "white"

        except LinAlgError:

            # Something went wrong inverting the covariance!
            for dist in [self.Size, self.Latitude, self.Contrast]:
                for slider in [dist.slider_mu, dist.slider_sigma]:
                    slider.bar_color = "firebrick"


class Distribution(object):
    def __init__(
        self, name, xmin, xmax, mu, sigma, pdf, gp_callback, throttle,
    ):
        # Store
        self.pdf = pdf
        self.gp_callback = gp_callback

        # Arrays
        x = np.linspace(xmin, xmax, 300)
        y = self.pdf(x, mu["value"], sigma["value"])
        self.source = ColumnDataSource(data=dict(x=x, y=y))

        # Plot them
        dx = (xmax - xmin) * 0.01
        self.plot = figure(
            plot_width=400,
            plot_height=400,
            sizing_mode="stretch_height",
            toolbar_location=None,
            x_range=(xmin - dx, xmax + dx),
            title="{} distribution".format(name),
        )
        self.plot.title.align = "center"
        self.plot.title.text_font_size = "14pt"
        self.plot.line(
            "x", "y", source=self.source, line_width=3, line_alpha=0.6
        )
        self.plot.xaxis.axis_label = name
        self.plot.xaxis.axis_label_text_font_style = "normal"
        self.plot.xaxis.axis_label_text_font_size = "12pt"
        self.plot.yaxis[0].formatter = FuncTickFormatter(code="return '  ';")
        self.plot.yaxis.axis_label = "probability"
        self.plot.yaxis.axis_label_text_font_style = "normal"
        self.plot.yaxis.axis_label_text_font_size = "12pt"
        self.plot.outline_line_width = 1
        self.plot.outline_line_alpha = 1
        self.plot.outline_line_color = "black"

        # Sliders
        self.slider_mu = Slider(
            start=mu["start"],
            end=mu["stop"],
            step=mu["step"],
            value=mu["value"],
            orientation="vertical",
            format="0.3f",
            css_classes=["custom-slider"],
            callback_policy="throttle",
            callback_throttle=throttle,
        )
        self.slider_mu.on_change("value_throttled", self.callback)
        self.slider_sigma = Slider(
            start=sigma["start"],
            end=sigma["stop"],
            step=sigma["step"],
            value=sigma["value"],
            orientation="vertical",
            format="0.3f",
            css_classes=["custom-slider"],
            name="sigma",
            callback_policy="throttle",
            callback_throttle=throttle,
        )
        self.slider_sigma.on_change("value_throttled", self.callback)

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
        self.layout = row(
            self.plot,
            column(svg_mu(), self.slider_mu, margin=(10, 10, 10, 10)),
            column(svg_sigma(), self.slider_sigma, margin=(10, 10, 10, 10)),
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

            # Update the mean and std. dev. lines
            self.mean_vline.location = self.slider_mu.value
            self.std_vline1.location = (
                self.slider_mu.value - self.slider_sigma.value
            )
            self.std_vline2.location = (
                self.slider_mu.value + self.slider_sigma.value
            )

        except AssertionError as e:

            # A param is out of bounds!
            if "std" in str(e):
                self.slider_sigma.bar_color = "firebrick"
            elif "mean" in str(e):
                self.slider_mu.bar_color = "firebrick"
            else:
                self.slider_sigma.bar_color = "firebrick"
                self.slider_mu.bar_color = "firebrick"

        finally:

            try:

                # Update the GP samples
                self.gp_callback(attr, old, new)

            except AssertionError as e:

                pass


class Application(object):
    def __init__(self, ydeg=15, npix=150, throttle=200):

        # The GP samples
        self.Samples = Samples(ydeg, npix, throttle)

        # The distributions
        self.Latitude = Distribution(
            "latitude [deg]",
            -90,
            90,
            params["latitude"]["mu"],
            params["latitude"]["sigma"],
            self.Samples.gp.latitude.transform.pdf,
            self.Samples.callback,
            throttle,
        )
        self.Size = Distribution(
            "size [deg]",
            0,
            90,
            params["size"]["mu"],
            params["size"]["sigma"],
            self.Samples.gp.size.transform.pdf,
            self.Samples.callback,
            throttle,
        )
        self.Contrast = Distribution(
            "contrast",
            -1,
            1,
            params["contrast"]["mu"],
            params["contrast"]["sigma"],
            self.Samples.gp.contrast.transform.pdf,
            self.Samples.callback,
            throttle,
        )

        # Tell the GP about the sliders
        self.Samples.Latitude = self.Latitude
        self.Samples.Size = self.Size
        self.Samples.Contrast = self.Contrast

        # Full layout
        self.layout = column(
            row(self.Latitude.layout, self.Size.layout, self.Contrast.layout),
            self.Samples.layout,
            style(),
        )

        # Add to the current document
        curdoc().add_root(self.layout)
        curdoc().title = "starry gaussian process"


Application()
