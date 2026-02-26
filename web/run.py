"""Entry point for the OSCE Grader web application."""

import logging
import os

from dotenv import load_dotenv

# Load .env from the web/ directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5100))

    if os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true"):
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        from waitress import serve
        logging.getLogger("osce_grader.web").info(
            "Starting OSCE Grader on http://0.0.0.0:%d", port
        )
        serve(app, host="0.0.0.0", port=port)
