#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# SKOPS pour la sérialisation
try:
    from skops import io as skio
    HAVE_SKOPS = True
except Exception:
    HAVE_SKOPS = False
    import joblib


def load_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print("⚠️ Impossible de lire Parquet sans pyarrow/fastparquet:", e, file=sys.stderr)
            raise
    elif path.suffix.lower() in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    else:
        raise ValueError(f"Format non supporté: {path.suffix}")


def find_target(df: pd.DataFrame) -> str:
    for c in ["ClassLabel", "label", "Label", "class", "target", "y"]:
        if c in df.columns:
            return c
    # fallback: objet avec peu de modalités
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in reversed(obj_cols):
        if df[c].nunique(dropna=True) <= max(20, int(0.05 * len(df))):
            return c
    raise RuntimeError("Colonne cible introuvable. Ajoutez --target <nom_colonne>.")


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    # compat: OneHotEncoder(sparse=False) fonctionne sur sklearn >=1.0 ;
    # éviter sparse_output (>=1.2) pour rétro-compatibilité.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def make_model(kind: str, random_state: int) -> object:
    kind = kind.lower()
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=random_state)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=random_state)
    if kind == "logreg":
        return LogisticRegression(max_iter=2000)
    raise ValueError(f"Modèle inconnu: {kind}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="Training.parquet")
    p.add_argument("--target", default=None)
    p.add_argument("--model", default="hgb", choices=["hgb", "rf", "logreg"], help="Algorithme principal")
    p.add_argument("--cv", type=int, default=0, help="n_splits CV (0 pour désactiver)")
    p.add_argument("--subsample", type=int, default=0, help="Échantillonner N lignes pour un entraînement rapide")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="student_model.skio")
    args = p.parse_args()

    rng = np.random.RandomState(args.seed)
    df = load_any(Path(args.train_file))

    target = args.target or find_target(df)
    y = df[target].astype("category")
    X = df.drop(columns=[target])

    if args.subsample and args.subsample < len(df):
        df_small = df.sample(n=args.subsample, random_state=args.seed, stratify=y if y.nunique() > 1 else None)
        y = df_small[target].astype("category")
        X = df_small.drop(columns=[target])
        print(f"➡️ Subsample utilisé: {len(df_small)} lignes")

    preprocess = build_preprocess(X)
    model = make_model(args.model, args.seed)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])

    # Cross‑validation (optionnelle, rapide si subsample)
    if args.cv and y.nunique() > 1:
        splits = max(2, min(args.cv, int(y.value_counts().min())))
        if splits < args.cv:
            warnings.warn(f"CV réduite à {splits} folds (classes rares)")
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | folds: {cv_scores}")

    # Hold‑out court pour un sanity check (20%)
    if y.nunique() > 1:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        acc = accuracy_score(y_va, pred)
        print(f"Hold‑out accuracy: {acc:.4f}")
        print(classification_report(y_va, pred))
    else:
        pipe.fit(X, y)

    # Ré‐entraînement sur tout le train (bonne pratique avant export)
    pipe.fit(X, y)

    # Export SKOPS
    out_path = Path(args.out)
    if HAVE_SKOPS:
        with open(out_path, "wb") as f:
            skio.dump(pipe, f)
        print(f"✅ Modèle exporté: {out_path}")
    else:
        alt = out_path.with_suffix(".pkl")
        joblib.dump(pipe, alt)
        print(f"⚠️ skops indisponible. Modèle sauvegardé en pickle: {alt}")

if __name__ == "__main__":
    main()
