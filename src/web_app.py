import argparse
import html
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd

from config import ASSETS_DIR, FEATURE_COLUMNS, MODEL_FILE, MODEL_METADATA_FILE


DEFAULT_SAMPLE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

FIELD_RANGES = {
    "fixed acidity": {"min": 4.0, "max": 16.5, "step": 0.1},
    "volatile acidity": {"min": 0.1, "max": 1.7, "step": 0.01},
    "citric acid": {"min": 0.0, "max": 1.1, "step": 0.01},
    "residual sugar": {"min": 0.5, "max": 16.0, "step": 0.1},
    "chlorides": {"min": 0.01, "max": 0.7, "step": 0.001},
    "free sulfur dioxide": {"min": 1, "max": 75, "step": 1},
    "total sulfur dioxide": {"min": 5, "max": 300, "step": 1},
    "density": {"min": 0.98, "max": 1.01, "step": 0.0001},
    "pH": {"min": 2.5, "max": 4.5, "step": 0.01},
    "sulphates": {"min": 0.2, "max": 2.2, "step": 0.01},
    "alcohol": {"min": 8.0, "max": 15.5, "step": 0.1},
}


def load_model_bundle() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_FILE}. Run: python src/train.py"
        )

    artifact = joblib.load(MODEL_FILE)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact

    return {
        "model": artifact,
        "model_name": "Trained Model",
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {},
    }


def load_metadata() -> dict:
    if MODEL_METADATA_FILE.exists():
        return json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
    return {}


MODEL_BUNDLE = load_model_bundle()
METADATA = load_metadata()


def parse_prediction_values(form_data: dict[str, list[str]]) -> dict[str, float]:
    values = {}
    for feature in MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS):
        field_name = feature.replace(" ", "_")
        raw_value = form_data.get(field_name, [DEFAULT_SAMPLE[feature]])[0]
        values[feature] = float(raw_value)
    return values


def predict_quality(values: dict[str, float]) -> float:
    feature_columns = MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS)
    sample = pd.DataFrame([values], columns=feature_columns)
    return float(MODEL_BUNDLE["model"].predict(sample)[0])


def metric_line() -> str:
    metrics = MODEL_BUNDLE.get("metrics") or METADATA.get("metrics") or {}
    if not metrics:
        return "Model is loaded and ready."

    return (
        f"MAE {float(metrics.get('mae', 0)):.4f} | "
        f"RMSE {float(metrics.get('rmse', 0)):.4f} | "
        f"R2 {float(metrics.get('r2', 0)):.4f}"
    )


def render_page(values: dict[str, float] | None = None, prediction: float | None = None) -> str:
    values = values or DEFAULT_SAMPLE
    model_name = html.escape(str(MODEL_BUNDLE.get("model_name", "Trained Model")))
    dataset_rows = METADATA.get("dataset_rows") or MODEL_BUNDLE.get("dataset_shape", [""])[0]

    inputs = []
    for feature in MODEL_BUNDLE.get("feature_columns", FEATURE_COLUMNS):
        field_name = feature.replace(" ", "_")
        label = html.escape(feature.title())
        value = html.escape(str(values.get(feature, DEFAULT_SAMPLE[feature])))
        field_range = FIELD_RANGES.get(feature, {"min": "", "max": "", "step": "any"})
        min_value = html.escape(str(field_range["min"]))
        max_value = html.escape(str(field_range["max"]))
        step_value = html.escape(str(field_range["step"]))
        inputs.append(
            f"""
            <label>
              <span>{label}</span>
              <input type="number" min="{min_value}" max="{max_value}" step="{step_value}" name="{field_name}" value="{value}" data-default="{html.escape(str(DEFAULT_SAMPLE[feature]))}" required>
            </label>
            """
        )

    result = ""
    if prediction is not None:
        if prediction >= 6.5:
            category = "High quality"
        elif prediction >= 5:
            category = "Good everyday wine"
        else:
            category = "Lower quality"

        result = f"""
        <section class="result">
          <div>
            <span>Predicted Quality</span>
            <small>{category}</small>
          </div>
          <strong>{prediction:.2f}</strong>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wine Quality Prediction</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #251921;
      --muted: #725d68;
      --line: rgba(119, 67, 82, 0.22);
      --accent: #9f1730;
      --accent-dark: #72101f;
      --gold: #d7a84f;
      --surface: rgba(255, 250, 246, 0.94);
      --page: #1c0c13;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(28, 12, 19, 0.82), rgba(28, 12, 19, 0.38)),
        url("/assets/wine-background.png?v=2") center / cover fixed,
        var(--page);
    }}

    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 38px;
    }}

    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 8px;
      padding: 24px;
      background:
        linear-gradient(135deg, rgba(255, 250, 246, 0.96), rgba(255, 243, 234, 0.86)),
        linear-gradient(90deg, rgba(159, 23, 48, 0.16), transparent);
      box-shadow: 0 22px 50px rgba(10, 3, 6, 0.32);
    }}

    .hero-copy {{
      min-width: 0;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 42px;
      line-height: 1.15;
      letter-spacing: 0;
      color: #24131a;
    }}

    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      font-size: 17px;
    }}

    .status {{
      text-align: right;
      min-width: 240px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
      padding: 14px 16px;
      border-left: 3px solid var(--gold);
      background: rgba(255, 255, 255, 0.54);
      border-radius: 6px;
    }}

    .visual-stack {{
      display: grid;
      gap: 12px;
      min-width: 0;
    }}

    .process-photo {{
      position: relative;
      margin: 0;
      height: 190px;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.42);
      box-shadow: 0 18px 38px rgba(54, 16, 27, 0.22);
      background: #2b1019;
    }}

    .process-photo img {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      filter: saturate(1.08) contrast(1.05);
    }}

    .process-photo::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, transparent 45%, rgba(24, 8, 14, 0.5));
      pointer-events: none;
    }}

    .status strong {{
      display: block;
      color: var(--ink);
      font-size: 16px;
    }}

    form {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 22px 55px rgba(10, 3, 6, 0.35);
      backdrop-filter: blur(6px);
    }}

    .model-info {{
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }}

    .info-panel {{
      padding: 18px;
      border: 1px solid rgba(255, 255, 255, 0.32);
      border-radius: 8px;
      background: rgba(255, 250, 246, 0.9);
      box-shadow: 0 18px 40px rgba(10, 3, 6, 0.2);
    }}

    .info-panel h2 {{
      margin: 0 0 8px;
      font-size: 20px;
      color: #2b121a;
      letter-spacing: 0;
    }}

    .info-panel p {{
      font-size: 15px;
    }}

    .metric-list {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}

    .metric-list div {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      padding: 9px 10px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.64);
      color: var(--muted);
      font-size: 14px;
    }}

    .metric-list strong {{
      color: var(--accent-dark);
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}

    label {{
      display: grid;
      gap: 6px;
      min-width: 0;
    }}

    label span {{
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }}

    input {{
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 15px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.92);
    }}

    input:focus {{
      outline: 2px solid rgba(143, 29, 44, 0.18);
      border-color: var(--accent);
    }}

    .actions {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-top: 18px;
    }}

    button {{
      height: 44px;
      border: 0;
      border-radius: 6px;
      padding: 0 18px;
      background: var(--accent);
      color: #fff;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(159, 23, 48, 0.28);
    }}

    button:hover {{
      background: var(--accent-dark);
    }}

    button.secondary-action {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      height: 44px;
      border: 1px solid rgba(159, 23, 48, 0.28);
      border-radius: 6px;
      padding: 0 16px;
      color: var(--accent-dark);
      background: rgba(255, 255, 255, 0.9);
      font-size: 15px;
      font-weight: 700;
      text-decoration: none;
      box-shadow: none;
      white-space: nowrap;
    }}

    button.secondary-action:hover {{
      color: #5b0d19;
      background: #fff0f2;
      border-color: rgba(159, 23, 48, 0.46);
    }}

    .button-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .result {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-top: 18px;
      padding: 16px;
      border: 1px solid #e2c4ca;
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(255, 250, 246, 0.98), rgba(255, 238, 232, 0.92));
      box-shadow: inset 4px 0 0 var(--gold);
    }}

    .result span {{
      display: block;
      color: var(--muted);
      font-size: 14px;
    }}

    .result small {{
      display: block;
      margin-top: 4px;
      color: var(--accent-dark);
      font-size: 16px;
      font-weight: 700;
    }}

    .result strong {{
      font-size: 34px;
      color: var(--accent);
    }}

    @media (max-width: 760px) {{
      header {{
        display: block;
      }}

      .status {{
        margin-top: 14px;
        text-align: left;
        border-left: 0;
        border-top: 3px solid var(--gold);
      }}

      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .model-info {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 560px) {{
      main {{
        width: min(100% - 20px, 1120px);
        padding: 18px 0;
      }}

      h1 {{
        font-size: 31px;
      }}

      form {{
        padding: 14px;
      }}

      .grid {{
        grid-template-columns: 1fr;
        gap: 10px;
      }}

      .actions {{
        display: block;
      }}

      button {{
        width: 100%;
      }}

      .button-row {{
        margin-top: 12px;
      }}

      .secondary-action {{
        width: 100%;
      }}

      .result {{
        margin-top: 14px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="hero-copy">
        <h1>Wine Quality Prediction</h1>
        <p>Enter wine chemistry values and predict the quality score using the trained model.</p>
      </div>
      <div class="visual-stack">
        <figure class="process-photo">
          <img src="/assets/prediction-process.jpeg?v=1" alt="Wine chemistry analysis visual">
        </figure>
        <div class="status">
          <strong>{model_name}</strong>
          {html.escape(metric_line())}<br>
          Dataset rows: {html.escape(str(dataset_rows))}
        </div>
      </div>
    </header>

    <section class="model-info">
      <article class="info-panel">
        <h2>What Random Forest Means</h2>
        <p>Random Forest is the trained model selected for this project. It combines many decision trees and averages their predictions, which usually gives a more stable wine-quality score than a single tree.</p>
      </article>
      <article class="info-panel">
        <h2>Model Performance</h2>
        <div class="metric-list">
          <div><span>MAE</span><strong>0.4296</strong></div>
          <div><span>RMSE</span><strong>0.5730</strong></div>
          <div><span>R2 Score</span><strong>0.4911</strong></div>
        </div>
      </article>
    </section>

    <form method="post" action="/">
      <div class="grid">
        {''.join(inputs)}
      </div>
      <div class="actions">
        <p>Model output is a predicted quality score, usually between 3 and 8 for this dataset.</p>
        <div class="button-row">
          <button class="secondary-action" type="button" id="reset-sample">Reset Sample Values</button>
          <button type="submit">Predict Quality</button>
        </div>
      </div>
      {result}
    </form>
  </main>
  <script>
    var resetButton = document.querySelector("#reset-sample");
    if (resetButton) {{
      resetButton.addEventListener("click", function () {{
        document.querySelectorAll("input[data-default]").forEach(function (input) {{
          input.value = input.dataset.default;
        }});

        var result = document.querySelector(".result");
        if (result) {{
          result.remove();
        }}
      }});
    }}
  </script>
</body>
</html>"""


class WinePredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        request_path = urlparse(self.path).path

        if request_path.startswith("/assets/"):
            self.respond_asset(request_path.removeprefix("/assets/"))
            return

        if request_path not in {"/", "/health"}:
            self.send_error(404)
            return

        if request_path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        self.respond_html(render_page())

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(body)

        try:
            values = parse_prediction_values(form_data)
            prediction = predict_quality(values)
            self.respond_html(render_page(values, prediction))
        except ValueError as exc:
            self.send_error(400, f"Invalid numeric input: {exc}")

    def respond_html(self, page: str) -> None:
        encoded = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_asset(self, asset_name: str) -> None:
        asset_path = (ASSETS_DIR / asset_name).resolve()
        assets_root = ASSETS_DIR.resolve()

        if assets_root not in asset_path.parents or not asset_path.exists():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        content = Path(asset_path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wine-quality prediction web app")
    default_port = int(os.environ.get("PORT", "8000"))
    default_host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WinePredictionHandler)
    print(f"Wine Quality Prediction app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
