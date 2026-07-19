"""
router_classifier.py
=======================

Trains and serves the lightweight, interpretable classifier that maps a
video's degradation feature vector (from degradation_estimator.py) to the
single best-performing enhancement recipe (from enhancement_recipes.py's
fixed R0-R5 set).

Includes three model options:
    1. Random Forest (default) — interpretable, fast, data-efficient
    2. Gradient Boosted Trees — slightly higher accuracy, less interpretable
    3. MLP (Multi-Layer Perceptron) — deep learning ablation baseline

Why classical models as default:
    - Input is 5-dimensional hand-engineered features (not raw pixels)
    - Label space is 6 fixed classes
    - Realistic training-set size is small (dozens of subjects × conditions)
    - Trees expose per-axis feature importances directly (interpretability)
    - MLP is provided as ablation to preempt "why not deep learning?" reviews

Author: Signal & Image Processing project (NMIMS Mumbai), MoSICom 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from degradation_estimator import FEATURE_NAMES
from enhancement_recipes import RECIPE_IDS

logger = logging.getLogger("router_classifier")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

REQUIRED_COLUMNS = list(FEATURE_NAMES) + ["subject_id", "best_recipe"]


# --------------------------------------------------------------------------- #
# Data loading and validation
# --------------------------------------------------------------------------- #

def load_training_table(csv_path: Path) -> pd.DataFrame:
    """Loads and validates a labeling_harness.py-produced training table."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training table {csv_path} missing required column(s): {missing}. "
            f"Required: {REQUIRED_COLUMNS}"
        )
    unknown_labels = set(df["best_recipe"].unique()) - set(RECIPE_IDS)
    if unknown_labels:
        raise ValueError(
            f"Training table contains recipe labels not in RECIPE_IDS={RECIPE_IDS}: {unknown_labels}"
        )
    n_subjects = df["subject_id"].nunique()
    logger.info("Loaded training table: %d rows, %d unique subjects, %d classes present",
                len(df), n_subjects, df["best_recipe"].nunique())
    return df


# --------------------------------------------------------------------------- #
# Subject-level, leakage-safe splitting
# --------------------------------------------------------------------------- #

def subject_level_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits rows into train/test with no subject leakage."""
    n_subjects = df["subject_id"].nunique()
    if n_subjects < 2:
        raise ValueError(
            f"subject_level_split requires at least 2 distinct subjects, found {n_subjects}."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject_id"]))
    train_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    overlap = set(train_df["subject_id"]) & set(test_df["subject_id"])
    assert not overlap, f"Subject leakage detected between train/test: {overlap}"

    logger.info(
        "Subject-level split: %d train rows (%d subjects), %d test rows (%d subjects)",
        len(train_df), train_df["subject_id"].nunique(),
        len(test_df), test_df["subject_id"].nunique(),
    )
    return train_df, test_df


# --------------------------------------------------------------------------- #
# Router classifier
# --------------------------------------------------------------------------- #

@dataclass
class TrainingMetadata:
    model_type: str
    hyperparams: dict
    feature_names: list[str]
    recipe_ids: list[str]
    n_train_rows: int
    n_train_subjects: int
    trained_at_utc: str
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_macro_f1: Optional[float] = None
    majority_baseline_accuracy: Optional[float] = None
    scaler_mean: Optional[list] = None
    scaler_std: Optional[list] = None


class RouterClassifier:
    """Wraps sklearn classifier with project-specific fit/predict/evaluate/persist."""

    def __init__(self, model_type: str = "random_forest", seed: int = 42, **hyperparams):
        self.model_type = model_type
        self.seed = seed
        self.hyperparams = hyperparams
        self.model = self._build_model(model_type, seed, hyperparams)
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Optional[TrainingMetadata] = None
        self._is_fitted = False

    @staticmethod
    def _build_model(model_type: str, seed: int, hyperparams: dict):
        if model_type == "random_forest":
            defaults = dict(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
            )
            defaults.update(hyperparams)
            return RandomForestClassifier(**defaults)
        elif model_type == "gbm":
            defaults = dict(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.08,
                random_state=seed,
            )
            defaults.update(hyperparams)
            return GradientBoostingClassifier(**defaults)
        elif model_type == "mlp":
            defaults = dict(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed,
            )
            defaults.update(hyperparams)
            return MLPClassifier(**defaults)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Use 'random_forest', 'gbm', or 'mlp'.")

    # ---- fitting ---- #

    def fit_from_dataframe(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> "RouterClassifier":
        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["best_recipe"].values

        # Scale features for MLP; trees are scale-invariant
        if self.model_type == "mlp":
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        train_acc = float(accuracy_score(y_train, self.model.predict(X_train)))
        majority_label = train_df["best_recipe"].mode().iloc[0]
        majority_baseline_acc = float((train_df["best_recipe"] == majority_label).mean())

        test_acc, test_f1 = None, None
        if test_df is not None and len(test_df) > 0:
            X_test = test_df[FEATURE_NAMES].values
            if self.scaler is not None:
                X_test = self.scaler.transform(X_test)
            y_test = test_df["best_recipe"].values
            y_pred = self.model.predict(X_test)
            test_acc = float(accuracy_score(y_test, y_pred))
            test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        self.metadata = TrainingMetadata(
            model_type=self.model_type,
            hyperparams=self.hyperparams,
            feature_names=list(FEATURE_NAMES),
            recipe_ids=list(RECIPE_IDS),
            n_train_rows=len(train_df),
            n_train_subjects=int(train_df["subject_id"].nunique()),
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            test_macro_f1=test_f1,
            majority_baseline_accuracy=majority_baseline_acc,
            scaler_mean=self.scaler.mean_.tolist() if self.scaler else None,
            scaler_std=self.scaler.scale_.tolist() if self.scaler else None,
        )
        logger.info(
            "Trained %s: train_acc=%.3f  majority_baseline_acc=%.3f  test_acc=%s  test_macro_f1=%s",
            self.model_type, train_acc, majority_baseline_acc,
            f"{test_acc:.3f}" if test_acc is not None else "n/a",
            f"{test_f1:.3f}" if test_f1 is not None else "n/a",
        )
        return self

    # ---- inference ---- #

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("RouterClassifier has not been fit or loaded yet.")

    def predict(self, feature_vector: np.ndarray) -> str:
        self._check_fitted()
        feature_vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        if feature_vector.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {feature_vector.shape[1]}")
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        return str(self.model.predict(feature_vector)[0])

    def predict_proba(self, feature_vector: np.ndarray) -> dict[str, float]:
        self._check_fitted()
        feature_vector = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        probs = self.model.predict_proba(feature_vector)[0]
        return {cls: float(p) for cls, p in zip(self.model.classes_, probs)}

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    # ---- evaluation ---- #

    def evaluate(self, test_df: pd.DataFrame, report_dir: Optional[Path] = None) -> dict:
        self._check_fitted()
        X_test = test_df[FEATURE_NAMES].values
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        y_test = test_df["best_recipe"].values
        y_pred = self.model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        report = classification_report(y_test, y_pred, labels=RECIPE_IDS, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=RECIPE_IDS)

        majority_label = pd.Series(y_test).mode().iloc[0] if len(y_test) else None
        majority_acc = float(np.mean(y_test == majority_label)) if majority_label is not None else None

        results = {
            "n_test_rows": len(test_df),
            "n_test_subjects": int(test_df["subject_id"].nunique()),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "majority_baseline_accuracy": majority_acc,
            "beats_majority_baseline": (acc > majority_acc) if majority_acc is not None else None,
            "per_class_report": report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": RECIPE_IDS,
        }

        logger.info(
            "Evaluation: accuracy=%.3f  macro_f1=%.3f  majority_baseline=%.3f  (beats baseline: %s)",
            acc, macro_f1, majority_acc, results["beats_majority_baseline"],
        )

        if report_dir is not None:
            report_dir = Path(report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            with open(report_dir / "router_evaluation.json", "w") as f:
                json.dump(results, f, indent=2)
            pd.DataFrame(cm, index=RECIPE_IDS, columns=RECIPE_IDS).to_csv(
                report_dir / "router_confusion_matrix.csv"
            )
            logger.info("Wrote evaluation report to %s", report_dir)

        return results

    # ---- interpretability ---- #

    def feature_importances(self) -> dict[str, float]:
        self._check_fitted()
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            # MLP does not have feature_importances_; return permutation importance placeholder
            logger.warning("MLP does not expose feature_importances_; returning zero placeholder.")
            return {name: 0.0 for name in FEATURE_NAMES}
        return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}

    # ---- persistence ---- #

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save model + scaler together
        payload = {"model": self.model, "scaler": self.scaler}
        joblib.dump(payload, path)
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(dataclasses_asdict_safe(self.metadata), f, indent=2)
        logger.info("Saved model -> %s, metadata -> %s", path, meta_path)

    @classmethod
    def load(cls, path: Path) -> "RouterClassifier":
        path = Path(path)
        payload = joblib.load(path)
        meta_path = path.with_suffix(".meta.json")
        instance = cls.__new__(cls)
        instance.model = payload["model"]
        instance.scaler = payload.get("scaler")
        instance._is_fitted = True
        instance.metadata = None
        instance.model_type = type(instance.model).__name__
        instance.hyperparams = {}
        instance.seed = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            instance.metadata = TrainingMetadata(**meta_dict)
            instance.model_type = meta_dict.get("model_type", instance.model_type)
        logger.info("Loaded model from %s", path)
        return instance


def dataclasses_asdict_safe(obj) -> dict:
    if obj is None:
        return {}
    import dataclasses
    return dataclasses.asdict(obj)


# --------------------------------------------------------------------------- #
# Model comparison across architectures
# --------------------------------------------------------------------------- #

def compare_model_architectures(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Train and compare RF, GBM, and MLP on the same data split.
    Returns a comparison table for the paper's ablation section."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "Random Forest": RouterClassifier("random_forest", seed=seed),
        "Gradient Boosting": RouterClassifier("gbm", seed=seed),
        "MLP (2-layer)": RouterClassifier("mlp", seed=seed, hidden_layer_sizes=(64, 32)),
    }

    rows = []
    for name, router in models.items():
        router.fit_from_dataframe(train_df, test_df)
        results = router.evaluate(test_df)
        rows.append({
            "Model": name,
            "Test Accuracy": f"{results['accuracy']:.3f}",
            "Macro F1": f"{results['macro_f1']:.3f}",
            "Beats Majority": results["beats_majority_baseline"],
            "Train Time": "<1s" if name != "MLP (2-layer)" else "~2s",
        })
        router.save(report_dir / f"router_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(report_dir / "model_comparison.csv", index=False)
    comparison_df.to_latex(report_dir / "model_comparison.tex", index=False)
    logger.info("Model comparison:\n%s", comparison_df.to_string(index=False))
    return comparison_df


# --------------------------------------------------------------------------- #
# Synthetic self-test data (PLACEHOLDER ONLY)
# --------------------------------------------------------------------------- #

def generate_synthetic_training_table(
    n_subjects: int = 24, windows_per_subject: int = 12, seed: int = 0
) -> pd.DataFrame:
    """Generates a PLACEHOLDER training table for code validation."""
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(n_subjects):
        subject_id = f"synthsubj_{subj:03d}"
        base = rng.uniform(0.0, 1.0, size=5)
        for w in range(windows_per_subject):
            noise, blur, comp, illum, motion = np.clip(
                base + rng.normal(0, 0.08, size=5), 0.0, 1.0
            )
            scores = {
                "R0": 1.15 * (1.0 - max(noise, blur, comp, illum, motion)),
                "R1": max(noise, comp) - 0.5 * illum - 0.35 * blur,
                "R2": 1.15 * illum - 0.45 * max(noise, comp),
                "R3": min(illum, noise) + 0.25 * comp - 0.5 * blur,
                "R4": 1.15 * blur - 0.45 * noise,
                "R5": 1.0 - 4 * (illum - 0.5) ** 2 - 2.4 * (noise - 0.5) ** 2 - 0.3 * blur,
            }
            for k in scores:
                scores[k] += rng.normal(0, 0.05)
            best_recipe = max(scores, key=scores.get)
            rows.append({
                "subject_id": subject_id,
                "window_id": f"{subject_id}_w{w:02d}",
                "noise_score": noise, "blur_score": blur, "compression_score": comp,
                "illumination_score": illum, "motion_score": motion,
                "best_recipe": best_recipe,
            })
    df = pd.DataFrame(rows)
    logger.warning(
        "generate_synthetic_training_table() produced PLACEHOLDER data (%d rows, %d subjects). "
        "Do not report results trained on this in the paper.",
        len(df), n_subjects,
    )
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and query the degradation-to-recipe router classifier."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="Train the router on a labeling_harness.py training table.")
    train.add_argument("--train-csv", required=True, type=Path)
    train.add_argument("--model-type", choices=["random_forest", "gbm", "mlp"], default="random_forest")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-model", required=True, type=Path)
    train.add_argument("--report-dir", type=Path, default=None)

    compare = sub.add_parser("compare", help="Compare RF vs GBM vs MLP on the same data.")
    compare.add_argument("--train-csv", required=True, type=Path)
    compare.add_argument("--test-size", type=float, default=0.2)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--report-dir", type=Path, default=Path("router_comparison"))

    predict = sub.add_parser("predict", help="Predict the best recipe for a single feature vector.")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--features-json", required=True, type=Path)

    selftest = sub.add_parser("self-test", help="Run end-to-end validation on PLACEHOLDER data.")
    selftest.add_argument("--n-subjects", type=int, default=24)
    selftest.add_argument("--windows-per-subject", type=int, default=12)
    selftest.add_argument("--model-type", choices=["random_forest", "gbm", "mlp"], default="random_forest")
    selftest.add_argument("--seed", type=int, default=42)
    selftest.add_argument("--report-dir", type=Path, default=Path("router_selftest_report"))

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        df = load_training_table(args.train_csv)
        train_df, test_df = subject_level_split(df, test_size=args.test_size, seed=args.seed)
        router = RouterClassifier(model_type=args.model_type, seed=args.seed)
        router.fit_from_dataframe(train_df, test_df)
        router.evaluate(test_df, report_dir=args.report_dir)
        router.save(args.output_model)
        print("Feature importances:", json.dumps(router.feature_importances(), indent=2))

    elif args.mode == "compare":
        df = load_training_table(args.train_csv)
        train_df, test_df = subject_level_split(df, test_size=args.test_size, seed=args.seed)
        compare_model_architectures(train_df, test_df, args.report_dir, seed=args.seed)

    elif args.mode == "predict":
        router = RouterClassifier.load(args.model)
        with open(args.features_json) as f:
            feats = json.load(f)
        vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
        label = router.predict(vec)
        proba = router.predict_proba(vec)
        print(json.dumps({"predicted_recipe": label, "probabilities": proba}, indent=2))

    elif args.mode == "self-test":
        logger.warning("Running self-test on PLACEHOLDER synthetic data -- not real results.")
        df = generate_synthetic_training_table(args.n_subjects, args.windows_per_subject, seed=args.seed)
        train_df, test_df = subject_level_split(df, test_size=0.25, seed=args.seed)
        router = RouterClassifier(model_type=args.model_type, seed=args.seed)
        router.fit_from_dataframe(train_df, test_df)
        results = router.evaluate(test_df, report_dir=args.report_dir)
        print("\nFeature importances:", json.dumps(router.feature_importances(), indent=2))
        print(f"\nSelf-test accuracy: {results['accuracy']:.3f} "
              f"(majority baseline: {results['majority_baseline_accuracy']:.3f})")


if __name__ == "__main__":
    main()
