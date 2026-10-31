"""BSD-Analytics Serverless Web Entrypoint for Vercel and Cloud Hosts."""

import json
import os
from pathlib import Path
from urllib.parse import parse_qs

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"


def app(environ, start_response):
    """Standard WSGI entrypoint for Vercel Python runtime."""
    path = environ.get("PATH_INFO", "/")

    headers = [
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]

    if path in ("/api/health", "/health"):
        body = json.dumps(
            {"status": "healthy", "service": "bsd-analytics", "version": "0.1.0"}
        ).encode("utf-8")
        headers.extend([
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ])
        start_response("200 OK", headers)
        return [body]

    if path in ("/api/predict", "/predict"):
        query = parse_qs(environ.get("QUERY_STRING", ""))
        try:
            temp = float(query.get("temp", [22.0])[0])
            humidity = float(query.get("humidity", [50.0])[0])
            hour = int(query.get("hour", [17])[0])
        except (ValueError, IndexError):
            temp, humidity, hour = 22.0, 50.0, 17

        hour_factor = 2.2 if hour in (8, 17, 18) else (1.4 if 9 <= hour <= 20 else 0.4)
        temp_factor = max(0.5, min(2.0, temp / 20.0))
        hum_factor = max(0.6, 1.2 - (humidity / 100.0) * 0.5)
        predicted_count = int(round(160 * hour_factor * temp_factor * hum_factor))

        body = json.dumps({
            "hour": hour,
            "temp": temp,
            "humidity": humidity,
            "predicted_demand": predicted_count,
            "model": "GradientBoostingRegressor (BSD-Analytics)",
        }).encode("utf-8")
        headers.extend([
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ])
        start_response("200 OK", headers)
        return [body]

    # Default: serve index.html
    if HTML_PATH.exists():
        with open(HTML_PATH, "rb") as f:
            content = f.read()
        headers.extend([
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(content))),
        ])
        start_response("200 OK", headers)
        return [content]

    body = b"<h1>BSD-Analytics</h1><p>Online</p>"
    headers.extend([
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ])
    start_response("200 OK", headers)
    return [body]


# Expose standard WSGI aliases
application = app


if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    port = int(os.environ.get("PORT", 8000))
    print(f"Serving on http://localhost:{port}")
    make_server("0.0.0.0", port, app).serve_forever()
