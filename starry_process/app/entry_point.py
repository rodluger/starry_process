from .app import app
from bokeh.server.server import Server


def entry_point():
    server = Server({"/": lambda doc: app.run(doc)})
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
