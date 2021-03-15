from .css import description, style, TEMPLATE, LOADING_SCREEN
from .design import get_intensity_design_matrix, get_flux_design_matrix
from .moll import get_latitude_lines, get_longitude_lines
from starry_process import StarryProcess
from starry_process.latitude import gauss2beta, beta2gauss
import numpy as np
from numpy.linalg import LinAlgError
from starry_process.compat import theano, tt
import sys
import os
import time
import bokeh
from bokeh.layouts import column, row, grid
from bokeh.models import (
    Div,
    ColumnDataSource,
    Slider,
    ColorBar,
    Label,
    RangeSlider,
    CustomJS,
    Button,
    Toggle,
    Span,
    HoverTool,
)
from bokeh.plotting import figure, curdoc
from bokeh.models.tickers import FixedTicker
from bokeh.models.formatters import FuncTickFormatter
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Plasma6, Category10
from bokeh.events import DocumentReady
from scipy.special import legendre as P
from scipy.stats import norm as Normal


# Parameter ranges & default values
params = {
    "latitude": {
        "mu": {
            "start": 0.0,
            "stop": 90.0,
            "step": 0.01,
            "value": 30.0,
            "label": "μ",
        },
        "sigma": {
            "start": 1.0,
            "stop": 45.0,
            "step": 0.01,
            "value": 5.0,
            "label": "σ",
        },
    },
    "size": {
        "r": {
            "start": 10.0,
            "stop": 45.0,
            "step": 0.1,
            "value": 20.0,
            "label": "r",
        }
    },
    "contrast": {
        "c": {
            "start": 0.01,
            "stop": 1.00,
            "step": 0.01,
            "value": 0.1,
            "label": "c",
        },
        "n": {
            "start": 1.00,
            "stop": 50.0,
            "step": 0.1,
            "value": 10.0,
            "label": "n",
        },
    },
}


def fluxnorm(x, **kwargs):
    return 1e3 * ((1 + x) / np.median(1 + x, **kwargs) - 1)


def spot_transform(ydeg, npts=1000, eps=1e-9, smoothing=0.075):
    theta = np.linspace(0, np.pi, npts)
    cost = np.cos(theta)
    B = np.hstack(
        [
            np.sqrt(2 * l + 1) * P(l)(cost).reshape(-1, 1)
            for l in range(ydeg + 1)
        ]
    )
    A = np.linalg.solve(B.T @ B + eps * np.eye(ydeg + 1), B.T)
    l = np.arange(ydeg + 1)
    i = l * (l + 1)
    S = np.exp(-0.5 * i * smoothing ** 2)
    A = S[:, None] * A
    return B @ A


class Samples(object):
    def __init__(
        self,
        ydeg,
        npix,
        npts,
        nmaps,
        throttle_time,
        nosmooth,
        gp,
        sample_function,
    ):
        # Settings
        self.ydeg = ydeg
        self.npix = npix
        self.npts = npts
        self.throttle_time = throttle_time
        self.nosmooth = nosmooth
        self.nmaps = nmaps
        self.gp = gp

        # Design matrices
        self.A_I = get_intensity_design_matrix(ydeg, npix)
        self.A_F = get_flux_design_matrix(ydeg, npts)

        def sample_ylm(r, mu_l, sigma_l, c, n):
            # Avoid issues at the boundaries
            if mu_l == 0:
                mu_l = 1e-2
            elif mu_l == 90:
                mu_l = 90 - 1e-2
            a, b = gauss2beta(mu_l, sigma_l)
            return sample_function(r, a, b, c, n)

        self.sample_ylm = sample_ylm

        # Draw three samples from the default distr
        self.ylm = self.sample_ylm(
            params["size"]["r"]["value"],
            params["latitude"]["mu"]["value"],
            params["latitude"]["sigma"]["value"],
            params["contrast"]["c"]["value"],
            params["contrast"]["n"]["value"],
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
            title="cmap",
        )
        self.slider.on_change("value", self.slider_callback)

        # Buttons
        self.seed_button = Button(
            label="re-seed",
            button_type="default",
            css_classes=["seed-button"],
            sizing_mode="fixed",
            height=30,
            width=75,
        )
        self.seed_button.on_click(self.seed_callback)

        self.smooth_button = Toggle(
            label="smooth",
            button_type="default",
            css_classes=["smooth-button"],
            sizing_mode="fixed",
            height=30,
            width=75,
            active=False,
        )
        self.smooth_button.disabled = bool(self.nosmooth)
        self.smooth_button.on_click(self.smooth_callback)

        self.auto_button = Toggle(
            label="auto",
            button_type="default",
            css_classes=["auto-button"],
            sizing_mode="fixed",
            height=30,
            width=75,
            active=True,
        )

        self.reset_button = Button(
            label="reset",
            button_type="default",
            css_classes=["reset-button"],
            sizing_mode="fixed",
            height=30,
            width=75,
        )
        self.reset_button.on_click(self.reset_callback)

        # Light curve samples
        self.flux_plot = [None for i in range(self.nmaps)]
        self.flux_source = [
            ColumnDataSource(
                data=dict(
                    xs=[np.linspace(0, 2, npts) for j in range(6)],
                    ys=[fluxnorm(self.A_F[j] @ self.ylm[i]) for j in range(6)],
                    color=[Plasma6[5 - j] for j in range(6)],
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
                plot_height=400,
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

        # Javascript callback to update light curves & images
        self.A_F_source = ColumnDataSource(data=dict(A_F=self.A_F))
        self.A_I_source = ColumnDataSource(data=dict(A_I=self.A_I))
        self.ylm_source = ColumnDataSource(data=dict(ylm=self.ylm))
        callback = CustomJS(
            args=dict(
                A_F_source=self.A_F_source,
                A_I_source=self.A_I_source,
                ylm_source=self.ylm_source,
                flux_source=self.flux_source,
                moll_source=self.moll_source,
            ),
            code="""
            var A_F = A_F_source.data['A_F'];
            var A_I = A_I_source.data['A_I'];
            var ylm = ylm_source.data['ylm'];
            var i, j, k, l, m, n;
            for (n = 0; n < {nmax}; n++) {{

                // Update the light curves
                var flux = flux_source[n].data['ys'];
                for (l = 0; l < {lmax}; l++) {{
                    for (m = 0; m < {mmax}; m++) {{
                        flux[l][m] = 0.0;
                        for (k = 0; k < {kmax}; k++) {{
                            flux[l][m] += A_F[{kmax} * ({mmax} * l + m) + k] * ylm[{kmax} * n + k];
                        }}
                    }}
                    // Normalize
                    var mean = flux[l].reduce((previous, current) => current += previous) / {mmax};
                    for (m = 0; m < {mmax}; m++) {{
                        flux[l][m] = 1e3 * ((1 + flux[l][m]) / (1 + mean) - 1)
                    }}
                }}
                flux_source[n].change.emit();

                // Update the images
                var image = moll_source[n].data['image'][0];
                for (i = 0; i < {imax}; i++) {{
                    for (j = 0; j < {jmax}; j++) {{
                        image[{jmax} * i + j] = 1.0;
                        for (k = 0; k < {kmax}; k++) {{
                            image[{jmax} * i + j] += A_I[{kmax} * ({jmax} * i + j) + k] * ylm[{kmax} * n + k];
                        }}
                    }}
                }}
                moll_source[n].change.emit();

            }}
            """.format(
                imax=self.npix,
                jmax=2 * self.npix,
                kmax=(self.ydeg + 1) ** 2,
                nmax=self.nmaps,
                lmax=self.A_F.shape[0],
                mmax=self.npts,
            ),
        )
        self.js_dummy = self.flux_plot[0].circle(x=0, y=0, size=1, alpha=0)
        self.js_dummy.glyph.js_on_change("size", callback)

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
        self.layout = grid([[self.plots]])

    def slider_callback(self, attr, old, new):
        self.color_mapper.low, self.color_mapper.high = self.slider.value

    def seed_callback(self, event):
        self.gp.random.seed(np.random.randint(0, 99999))
        tmp = self.auto_button.active
        self.callback(None, None, None)
        self.auto_button.active = tmp

    def reset_callback(self, event):
        self.Size.sliders[0].value = params["size"]["r"]["value"]
        self.Latitude.sliders[0].value = params["latitude"]["mu"]["value"]
        self.Latitude.sliders[1].value = params["latitude"]["sigma"]["value"]
        self.Contrast.sliders[0].value = params["contrast"]["c"]["value"]
        self.Contrast.sliders[1].value = params["contrast"]["n"]["value"]
        self.Size.callback(None, None, None)
        self.Latitude.callback(None, None, None)
        self.gp.random.seed(0)
        self.slider.value = (0.5, 1.2)
        self.slider_callback(None, None, None)
        self.smooth_button.active = False
        self.smooth_callback(None)
        self.auto_button.active = True
        self.callback(None, None, None)

    def smooth_callback(self, event):
        if self.smooth_button.active:
            self.Latitude.throttle_time = self.throttle_time
            self.Size.throttle_time = self.throttle_time
            self.Contrast.throttle_time = self.throttle_time
        else:
            self.Latitude.throttle_time = 0
            self.Size.throttle_time = 0
            self.Contrast.throttle_time = 0

    def callback(self, attr, old, new):
        try:

            if self.auto_button.active:
                self.gp.random.seed(np.random.randint(0, 99999))

            # Draw the samples
            self.ylm = self.sample_ylm(
                self.Size.sliders[0].value,
                self.Latitude.sliders[0].value,
                self.Latitude.sliders[1].value,
                self.Contrast.sliders[0].value,
                self.Contrast.sliders[1].value,
            )[0]

            # HACK: Trigger the JS callback by modifying
            # a property of the `js_dummy` glyph
            self.ylm_source.data["ylm"] = self.ylm
            self.js_dummy.glyph.size = 3 - self.js_dummy.glyph.size

            # If everything worked, ensure the sliders are active
            for slider in (
                self.Size.sliders
                + self.Latitude.sliders
                + self.Contrast.sliders
            ):
                slider.bar_color = "white"

        except Exception as e:

            # Something went wrong inverting the covariance!
            for slider in (
                self.Size.sliders
                + self.Latitude.sliders
                + self.Contrast.sliders
            ):
                slider.bar_color = "firebrick"

            print("An error occurred when computing the covariance:")
            print(e)


class Integral(object):
    def __init__(
        self,
        params,
        gp_callback,
        limits=(-90, 90),
        xticks=[-90, -60, -30, 0, 30, 60, 90],
        funcs=[],
        labels=[],
        xlabel="",
        ylabel="",
        distribution=False,
        legend_location="bottom_right",
        npts=300,
    ):
        # Store
        self.params = params
        self.limits = limits
        self.funcs = funcs
        self.gp_callback = gp_callback
        self.last_run = 0.0
        self.throttle_time = 0.0
        self.distribution = distribution

        # Arrays
        if len(self.funcs):
            xmin, xmax = self.limits
            xs = [np.linspace(xmin, xmax, npts) for f in self.funcs]
            ys = [
                f(x, *[self.params[p]["value"] for p in self.params.keys()])
                for x, f in zip(xs, self.funcs)
            ]
            colors = Category10[10][: len(xs)]
            lws = np.append([3], np.ones(10))[: len(xs)]
            self.source = ColumnDataSource(
                data=dict(xs=xs, ys=ys, colors=colors, lws=lws)
            )

            # Plot them
            dx = (xmax - xmin) * 0.01
            self.plot = figure(
                plot_width=400,
                plot_height=600,
                toolbar_location=None,
                x_range=(xmin - dx, xmax + dx),
                title=xlabel,
                sizing_mode="stretch_both",
            )
            self.plot.title.align = "center"
            self.plot.title.text_font_size = "14pt"
            self.plot.multi_line(
                xs="xs",
                ys="ys",
                line_color="colors",
                source=self.source,
                line_width="lws",
                line_alpha=0.6,
            )
            self.plot.xaxis.axis_label_text_font_style = "normal"
            self.plot.xaxis.axis_label_text_font_size = "12pt"
            self.plot.xaxis.ticker = FixedTicker(ticks=xticks)
            self.plot.yaxis[0].formatter = FuncTickFormatter(
                code="return '  ';"
            )
            self.plot.yaxis.axis_label = ylabel
            self.plot.yaxis.axis_label_text_font_style = "normal"
            self.plot.yaxis.axis_label_text_font_size = "12pt"
            self.plot.outline_line_width = 1
            self.plot.outline_line_alpha = 1
            self.plot.outline_line_color = "black"
            self.plot.toolbar.active_drag = None
            self.plot.toolbar.active_scroll = None
            self.plot.toolbar.active_tap = None

            # Legend
            for j, label in enumerate(labels):
                self.plot.line(
                    [0, 0],
                    [0, 0],
                    legend_label=label,
                    line_color=Category10[10][j],
                )
            self.plot.legend.location = legend_location
            self.plot.legend.title_text_font_style = "bold"
            self.plot.legend.title_text_font_size = "8pt"
            self.plot.legend.label_text_font_size = "8pt"
            self.plot.legend.spacing = 0
            self.plot.legend.label_height = 5
            self.plot.legend.glyph_height = 15

        else:

            self.plot = None

        # Sliders
        self.sliders = []
        for p in self.params.keys():
            slider = Slider(
                start=self.params[p]["start"],
                end=self.params[p]["stop"],
                step=self.params[p]["step"],
                value=self.params[p]["value"],
                orientation="horizontal",
                format="0.3f",
                css_classes=["custom-slider"],
                sizing_mode="stretch_width",
                height=10,
                show_value=False,
                title=self.params[p]["label"],
            )
            slider.on_change("value_throttled", self.callback)
            slider.on_change("value", self.callback_throttled)
            self.sliders.append(slider)

        # HACK: Add a hidden slider to get the correct
        # amount of padding below the graph
        if len(self.params.keys()) < 2:
            slider = Slider(
                start=0,
                end=1,
                step=0.1,
                value=0.5,
                orientation="horizontal",
                css_classes=["custom-slider", "hidden-slider"],
                sizing_mode="stretch_width",
                height=10,
                show_value=False,
                title="s",
            )
            self.hidden_sliders = [slider]
        else:
            self.hidden_sliders = []

        # Show mean and std. dev.?
        if self.distribution:
            self.mean_vline = Span(
                location=self.sliders[0].value,
                dimension="height",
                line_color="black",
                line_width=1,
                line_dash="dashed",
            )
            self.std_vline1 = Span(
                location=self.sliders[0].value - self.sliders[1].value,
                dimension="height",
                line_color="black",
                line_width=1,
                line_dash="dotted",
            )
            self.std_vline2 = Span(
                location=self.sliders[0].value + self.sliders[1].value,
                dimension="height",
                line_color="black",
                line_width=1,
                line_dash="dotted",
            )
            self.plot.renderers.extend(
                [self.mean_vline, self.std_vline1, self.std_vline2]
            )

        # Full layout
        if self.plot is not None:
            self.layout = grid(
                [
                    [self.plot],
                    [
                        column(
                            *self.sliders,
                            *self.hidden_sliders,
                            sizing_mode="stretch_width",
                        )
                    ],
                ]
            )
        else:
            self.layout = grid(
                [
                    [
                        column(
                            *self.sliders,
                            *self.hidden_sliders,
                            sizing_mode="stretch_width",
                        )
                    ]
                ]
            )

    def callback(self, attr, old, new):
        try:

            # Update the plot
            if len(self.funcs):

                self.source.data["ys"] = [
                    f(x, *[slider.value for slider in self.sliders])
                    for x, f in zip(self.source.data["xs"], self.funcs)
                ]
                for slider in self.sliders:
                    slider.bar_color = "white"
                self.plot.background_fill_color = "white"
                self.plot.background_fill_alpha = 1

                # Update the mean and std. dev. lines
                if self.distribution:
                    self.mean_vline.location = self.sliders[0].value
                    self.std_vline1.location = (
                        self.sliders[0].value - self.sliders[1].value
                    )
                    self.std_vline2.location = (
                        self.sliders[0].value + self.sliders[1].value
                    )

        except Exception as e:

            # A param is out of bounds!
            for slider in self.sliders:
                slider.bar_color = "firebrick"
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
    def __init__(
        self,
        ydeg=15,
        npix=100,
        npts=300,
        nmaps=5,
        throttle_time=0.40,
        load_timeout=1.0,
        nosmooth=False,
    ):
        self.ydeg = ydeg
        self.npix = npix
        self.npts = npts
        self.nmaps = nmaps
        self.throttle_time = throttle_time
        self.load_timeout = load_timeout
        self.nosmooth = nosmooth
        self._compiled = False

    def compile(self):

        # Compile the GP
        print("Compiling. This may take up to one minute...")
        r = tt.dscalar()
        a = tt.dscalar()
        b = tt.dscalar()
        c = tt.dscalar()
        n = tt.dscalar()
        self.gp = StarryProcess(ydeg=self.ydeg, r=r, a=a, b=b, c=c, n=n)
        self.gp.random.seed(238)
        self.sample_function = theano.function(
            [r, a, b, c, n],
            [self.gp.sample_ylm(nsamples=self.nmaps)],
            no_default_updates=True,
        )
        self._compiled = True
        print("Done!")

    def run(self, doc=None):

        # Get current document if needed
        if doc is None:
            doc = curdoc()
        doc.title = "starry process"
        doc.template = TEMPLATE

        # Show the loading screen
        self.layout = Div(text=LOADING_SCREEN, css_classes=["body-wrapper"])
        doc.add_root(self.layout)

        # Set a delay timer for safety; when it's done, load the widgets
        doc.add_timeout_callback(
            lambda: self._run(doc), int(1000 * self.load_timeout)
        )

    def _run(self, doc=None):

        # Get current document if needed
        if doc is None:
            doc = curdoc()

        # Compile if needed
        if not self._compiled:
            self.compile()

        print("Rendering...")

        # The GP samples
        self.Samples = Samples(
            self.ydeg,
            self.npix,
            self.npts,
            self.nmaps,
            self.throttle_time,
            self.nosmooth,
            self.gp,
            self.sample_function,
        )

        # The integrals
        sp = StarryProcess(ydeg=self.ydeg)
        pdf = lambda x, mu, sigma: sp.latitude._pdf(x, *gauss2beta(mu, sigma))
        pdf_gauss = lambda x, mu, sigma: 0.5 * (
            Normal.pdf(x, -mu, sigma) + Normal.pdf(x, mu, sigma)
        )
        npts = 300
        T = spot_transform(self.ydeg, npts)
        self.Latitude = Integral(
            params["latitude"],
            self.Samples.callback,
            funcs=[pdf, pdf_gauss],
            labels=["pdf", "laplace"],
            xlabel="latitude distribution",
            ylabel="probability",
            distribution=True,
            legend_location="top_left",
        )
        self.Size = Integral(
            params["size"],
            self.Samples.callback,
            funcs=[
                lambda x, r: T
                @ (1 / (1 + np.exp(-300 * np.pi / 180 * (np.abs(x) - r))) - 1),
                lambda x, r: (
                    1 / (1 + np.exp(-300 * np.pi / 180 * (np.abs(x) - r))) - 1
                ),
            ],
            labels=["ylm", "true"],
            xlabel="spot profile",
            ylabel="intensity",
            npts=npts,
        )
        self.Contrast = Integral(params["contrast"], self.Samples.callback)

        # Tell the GP about the sliders
        self.Samples.Latitude = self.Latitude
        self.Samples.Size = self.Size
        self.Samples.Contrast = self.Contrast

        # Settings
        ControlPanel = column(
            Div(text="<h1>settings</h1>", css_classes=["control-title"]),
            self.Contrast.layout,
            self.Samples.slider,
            row(
                self.Samples.smooth_button,
                self.Samples.auto_button,
                self.Samples.seed_button,
                self.Samples.reset_button,
                sizing_mode="scale_both",
                css_classes=["button-row"],
            ),
            Div(text=description, css_classes=["control-description"]),
            sizing_mode="scale_both",
        )

        # Full layout
        layout = column(
            row(
                self.Latitude.layout,
                self.Size.layout,
                ControlPanel,
                sizing_mode="scale_both",
            ),
            self.Samples.layout,
            style(),
            sizing_mode="scale_both",
        )

        print("Done rendering.")

        # Remove the loading screen
        doc.remove_root(self.layout)

        # Add the interface
        self.layout = layout
        doc.add_root(self.layout)


# Instantiate the base app
app = Application(
    ydeg=int(os.getenv("YDEG", 15)),
    npix=int(os.getenv("NPIX", 100)),
    npts=int(os.getenv("NPTS", 300)),
    nmaps=int(os.getenv("NMAPS", 5)),
    nosmooth=int(os.getenv("NOSMOOTH", 0)),
    throttle_time=float(os.getenv("THROTTLE_TIME", 0.15)),
    load_timeout=float(os.getenv("LOAD_TIMEOUT", 1.0)),
)
