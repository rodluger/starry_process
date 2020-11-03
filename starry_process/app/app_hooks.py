from .app import app


def on_server_loaded(server_context):
    app.compile()
