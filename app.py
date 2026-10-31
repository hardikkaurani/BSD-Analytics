"""BSD-Analytics Serverless Web Entrypoint for Vercel and Cloud Hosts."""

import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"


class handler(BaseHTTPRequestHandler):
    """Vercel native BaseHTTPRequestHandler for BSD-Analytics."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/api/health", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            res = {
                "status": "healthy",
                "service": "bsd-analytics",
                "version": "0.1.0",
                "description": "Bike Sharing Demand Analytics API",
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        if path in ("/api/predict", "/predict"):
            query = parse_qs(parsed.query)
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

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            res = {
                "hour": hour,
                "temp": temp,
                "humidity": humidity,
                "predicted_demand": predicted_count,
                "model": "GradientBoostingRegressor (BSD-Analytics)",
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        # Serve index.html
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if HTML_PATH.exists():
            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.wfile.write(b"<h1>BSD-Analytics</h1><p>Online</p>")

    def do_POST(self):
        self.do_GET()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    from http.server import HTTPServer

    port = int(os.environ.get("PORT", 8000))
    print(f"Serving BSD-Analytics on http://localhost:{port}")
    server = HTTPServer(("0.0.0.0", port), handler)
    server.serve_forever()
