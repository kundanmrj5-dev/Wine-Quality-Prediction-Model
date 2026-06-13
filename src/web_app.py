import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import joblib
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_FILE, MODEL_METADATA_FILE


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
        inputs.append(
            f"""
            <label>
              <span>{label}</span>
              <input type="number" step="any" name="{field_name}" value="{value}" required>
            </label>
            """
        )

    result = ""
    if prediction is not None:
        result = f"""
        <section class="result">
          <span>Predicted Quality</span>
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
      --ink: #1f2933;
      --muted: #627386;
      --line: #d8dee6;
      --accent: #8f1d2c;
      --accent-dark: #6f1522;
      --surface: #ffffff;
      --page: #f6f3ef;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--page);
    }}

    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }}

    header {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
      line-height: 1.15;
      letter-spacing: 0;
    }}

    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}

    .status {{
      text-align: right;
      min-width: 240px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
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
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.08);
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
    }}

    input {{
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font-size: 15px;
      color: var(--ink);
      background: #fff;
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
    }}

    button:hover {{
      background: var(--accent-dark);
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
      background: #fff7f8;
    }}

    .result span {{
      color: var(--muted);
      font-size: 14px;
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
      }}

      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 560px) {{
      main {{
        width: min(100% - 20px, 1120px);
        padding: 18px 0;
      }}

      h1 {{
        font-size: 28px;
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

      .result {{
        margin-top: 14px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Wine Quality Prediction</h1>
        <p>Enter wine chemistry values and predict the quality score using the trained model.</p>
      </div>
      <div class="status">
        <strong>{model_name}</strong>
        {html.escape(metric_line())}<br>
        Dataset rows: {html.escape(str(dataset_rows))}
      </div>
    </header>

    <form method="post" action="/">
      <div class="grid">
        {''.join(inputs)}
      </div>
      <div class="actions">
        <p>Model output is a predicted quality score, usually between 3 and 8 for this dataset.</p>
        <button type="submit">Predict Quality</button>
      </div>
      {result}
    </form>
  </main>
</body>
</html>"""


class WinePredictionHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/", "/health"}:
            self.send_error(404)
            return

        if self.path == "/health":
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

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wine-quality prediction web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WinePredictionHandler)
    print(f"Wine Quality Prediction app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
