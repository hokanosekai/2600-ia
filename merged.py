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
    """Charge les données depuis un fichier Parquet ou CSV."""
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
    """Trouve la colonne cible probable dans le DataFrame."""
    for c in ["ClassLabel", "label", "Label", "class", "target", "y"]:
        if c in df.columns:
            return c
    # Fallback: colonne de type objet avec peu de valeurs uniques
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in reversed(obj_cols):
        if df[c].nunique(dropna=True) <= max(20, int(0.05 * len(df))):
            return c
    raise RuntimeError("Colonne cible introuvable. Ajoutez --target <nom_colonne>.")


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le pipeline de prétraitement pour les colonnes numériques et catégorielles."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="passthrough")


def make_model(kind: str, random_state: int) -> object:
    """Crée une instance de modèle de classification."""
    kind = kind.lower()
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=random_state)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=random_state)
    if kind == "logreg":
        return LogisticRegression(max_iter=2000)
    raise ValueError(f"Modèle inconnu: {kind}")


def main():
    """Fonction principale pour l'entraînement et l'export du modèle."""
    p = argparse.ArgumentParser(description="Script d'entraînement de modèle de classification.")
    p.add_argument("--train-file", default="Training.parquet", help="Fichier de données d'entraînement (Parquet ou CSV).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible (détectée automatiquement si non fournie).")
    p.add_argument("--model", default="hgb", choices=["hgb", "rf", "logreg"], help="Algorithme principal à utiliser.")
    p.add_argument("--cv", type=int, default=0, help="Nombre de folds pour la validation croisée (0 pour désactiver).")
    p.add_argument("--subsample", type=int, default=0, help="Échantillonner N lignes pour un entraînement rapide (0 pour tout utiliser).")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité.")
    p.add_argument("--out", default="student_model.skio", help="Chemin du fichier de sortie pour le modèle sérialisé.")
    args = p.parse_args()

    print("--- Étape 1: Chargement et Nettoyage des Données ---")
    df = load_any(Path(args.train_file))
    df.columns = df.columns.str.strip()
    
    # Remplacer les valeurs infinies par NaN, qui seront traitées par l'imputer
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Données chargées: {len(df)} lignes.")

    print("\n--- Étape 2: Préparation des Données ---")
    target_col = args.target or find_target(df)
    y = df[target_col].astype("category")
    X = df.drop(columns=[target_col])
    print(f"Colonne cible '{target_col}' identifiée.")

    if args.subsample and args.subsample < len(df):
        df_small = df.sample(n=args.subsample, random_state=args.seed)
        y = df_small[target_col].astype("category")
        X = df_small.drop(columns=[target_col])
        print(f"➡️ Sous-échantillon de {len(df_small)} lignes utilisé pour l'entraînement.")

    print("\n--- Étape 3: Création du Pipeline ---")
    preprocess_pipeline = build_preprocess(X)
    model = make_model(args.model, args.seed)
    pipe = Pipeline([("preprocess", preprocess_pipeline), ("model", model)])
    print("Pipeline créé avec succès:")
    print(pipe)

    # Validation croisée (optionnelle)
    if args.cv > 0 and y.nunique() > 1:
        print(f"\n--- Étape 4a: Validation Croisée ({args.cv} folds) ---")
        splits = max(2, min(args.cv, int(y.value_counts().min())))
        if splits < args.cv:
            warnings.warn(f"CV réduite à {splits} folds en raison de classes peu représentées.")
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Folds: {cv_scores}")

    # Évaluation sur un jeu de test (hold-out)
    if y.nunique() > 1:
        print("\n--- Étape 4b: Évaluation sur Jeu de Test (Hold-out) ---")
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        acc = accuracy_score(y_va, pred)
        print(f"Hold-out Accuracy: {acc:.4f}")
        print("Rapport de classification:")
        print(classification_report(y_va, pred))
    else:
        # Pas de validation possible si une seule classe
        pipe.fit(X, y)

    print("\n--- Étape 5: Entraînement Final et Export ---")
    # Ré-entraînement sur l'ensemble des données pour le modèle final
    pipe.fit(X, y)
    print("Modèle final entraîné sur l'ensemble des données.")

    # Export du modèle
    out_path = Path(args.out)
    if HAVE_SKOPS:
        with open(out_path, "wb") as f:
            skio.dump(pipe, f)
        print(f"✅ Modèle exporté avec skops: {out_path}")
    else:
        alt_path = out_path.with_suffix(".pkl")
        joblib.dump(pipe, alt_path)
        print(f"⚠️ skops non disponible. Modèle sauvegardé avec joblib: {alt_path}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        # Ignorer certains avertissements pour une sortie plus propre
        warnings.filterwarnings('ignore', category=UserWarning, module='skops')
        warnings.filterwarnings('ignore', category=FutureWarning)
        main()
