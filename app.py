"""BSD-Analytics Serverless Web Entrypoint."""

from main import app, application

__all__ = ["app", "application"]

if __name__ == "__main__":
    import os
    from wsgiref.simple_server import make_server

    port = int(os.environ.get("PORT", 8000))
    print(f"Serving on http://localhost:{port}")
    make_server("0.0.0.0", port, app).serve_forever()
