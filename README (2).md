# Research Sandbox Deployment

A local API that serves the treatment outcome model for testing, demonstrations, and
teaching how models are packaged as services.

> **Research sandbox only.** The model is trained on synthetic data and has not been
> validated on real patients. Do not use it for diagnosis, treatment, or any clinical
> decision. Do not expose this service to the public internet, and do not connect it to
> patient records. Real clinical deployment requires external validation, clinical review,
> ethics and data-governance approval, and regulatory clearance where applicable.

## Files

```
deployment/
├── train_and_export.py     # Trains the model on the synthetic cohort, saves artifacts/model.joblib
├── app.py                  # FastAPI service: /health and /predict
├── test_app.py             # Tests for the API
├── requirements-deploy.txt
├── artifacts/              # Created by train_and_export.py
└── README.md
```

## Setup

From the `medical-treatment-analysis/deployment` folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-deploy.txt
```

## Steps

1. Train and save the model:
   ```bash
   python train_and_export.py
   ```
2. Run the tests:
   ```bash
   pytest -q
   ```
3. Start the sandbox server on your own machine only:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```
4. Open `http://127.0.0.1:8000/docs` to try the API in the browser, or send a request:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"age":60,"bmi":28,"systolic_bp":135,"biomarker":1.2,"comorbidities":2,"treatment":1}'
   ```

## Endpoints

| Method | Path       | Purpose                                                   |
|--------|------------|-----------------------------------------------------------|
| GET    | `/health`  | Service status, model metadata, disclaimer                |
| POST   | `/predict` | Returns predicted probability and label with disclaimer   |

Every prediction response includes `"research_only": true` and the disclaimer text.

## Why `127.0.0.1`

Binding to `127.0.0.1` means only your own computer can reach the server. Binding to `0.0.0.0`
would make it reachable by anyone on your network. Keep the sandbox on localhost.

## Limitations

- Trained on synthetic data, so predictions carry no clinical meaning.
- No authentication, logging, monitoring, or audit trail. A real service would need all of these.
- No versioned model registry or retraining pipeline.
- The `/predict` input validation checks ranges, not clinical plausibility.
