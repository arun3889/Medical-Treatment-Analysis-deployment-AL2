"""
Train the research model on the SYNTHETIC demo cohort and save it as a joblib artifact.

RESEARCH SANDBOX ONLY. This script uses no real patient data.

Usage (from the deployment/ folder):
  python train_and_export.py
"""

import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the synthetic cohort generator from the analysis project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from treatment_analysis import RANDOM_STATE, make_synthetic_cohort  # noqa: E402

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
OUTCOME = "recovered"
C_VALUE = 0.1  # best logistic regression C from the analysis run


def main():
    frame = make_synthetic_cohort(seed=RANDOM_STATE)
    X = frame.drop(columns=[OUTCOME])
    y = frame[OUTCOME]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C_VALUE, max_iter=2000, class_weight="balanced"),
    )
    model.fit(X, y)

    bundle = {
        "model": model,
        "features": list(X.columns),
        "metadata": {
            "model_type": "logistic_regression",
            "trained_on": "synthetic demo cohort (no real patient data)",
            "version": "0.1.0-research",
            "intended_use": "research and education only; not for clinical use",
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")
    print(f"Features: {bundle['features']}")


if __name__ == "__main__":
    main()
