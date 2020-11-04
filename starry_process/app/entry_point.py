from .app import app
from bokeh.server.server import Server


def entry_point():
    """
    This is the entry point for the command-line script `starry-process`.
    We set up a server and launch the page below.
    
    """
    server = Server({"/": lambda doc: app.run(doc)})
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
