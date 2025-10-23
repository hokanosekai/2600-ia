#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys
import warnings
from datetime import datetime

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

# Imbalanced-learn (resampling)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

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

    # OneHotEncoder returns sparse by default. SMOTE operates on dense arrays.
    # Force dense output from the ColumnTransformer with sparse_threshold=0.
    # (We keep OneHotEncoder default; ColumnTransformer will densify.)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ], remainder="passthrough", sparse_threshold=0)


def make_model(kind: str, random_state: int) -> object:

    """Crée une instance de modèle de classification."""
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=1000,
            learning_rate=0.05,
            max_depth=12,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
            class_weight='balanced'
        )
    elif kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            # max_depth=15,
            # min_samples_split=10,
            # min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
    elif kind == "xgb":  # Ajouter XGBoost
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            scale_pos_weight=1
        )
    raise ValueError(f"Modèle inconnu: {kind}")


def balance_df(
    df: pd.DataFrame,
    target_col: str,
    per_class: int,
    mode: str,
    seed: int,
    include_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Retourne un DataFrame équilibré à per_class par classe.
    mode:
      - 'downsample': réduit les classes > per_class, ne duplique pas les classes < per_class (elles restent < per_class)
      - 'upsample': duplique (avec remplacement) les classes < per_class, ne réduit pas celles > per_class
      - 'both': downsample au-dessus, upsample en-dessous
    """
    rng = np.random.RandomState(seed)
    if include_classes is None:
        classes = df[target_col].astype("category").cat.categories.tolist()
    else:
        classes = [c for c in include_classes if c in df[target_col].unique()]

    parts = []
    for cls in classes:
        block = df[df[target_col] == cls]
        n = len(block)

        # Déterminer la taille cible pour cette classe
        target_n = per_class
        if mode == "downsample":
            target_n = min(per_class, n)
        elif mode == "upsample":
            target_n = max(per_class, n)
        elif mode == "both":
            target_n = per_class

        if target_n <= n:
            # échantillonnage sans remplacement
            sampled = block.sample(n=target_n, replace=False, random_state=seed)
        else:
            # sur-échantillonnage avec remplacement (duplication)
            idx = rng.choice(block.index.values, size=target_n, replace=True)
            sampled = block.loc[idx]

        parts.append(sampled)

    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced


def main():
    """Fonction principale pour l'entraînement et l'export du modèle."""
    p = argparse.ArgumentParser(description="Script d'entraînement de modèle de classification.")
    p.add_argument("--train-file", default="Training.parquet", help="Fichier de données d'entraînement (Parquet ou CSV).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible (détectée automatiquement si non fournie).")
    p.add_argument("--model", default="hgb", choices=["hgb", "rf", "xgb"], help="Algorithme principal à utiliser.")
    p.add_argument("--cv", type=int, default=0, help="Nombre de folds pour la validation croisée (0 pour désactiver).")
    p.add_argument("--subsample", type=int, default=0, help="Échantillonner N lignes pour un entraînement rapide (0 pour tout utiliser).")
    p.add_argument("--testsample", type=float, default=0.2, help="Fraction des données à utiliser comme jeu de test hold-out (0 pour désactiver).")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour la reproductibilité.")
    p.add_argument(
        "--scoring",
        default="accuracy",
        choices=["accuracy", "f1_macro", "balanced_accuracy", "f1_weighted"],
        help="Métrique utilisée pour la validation croisée.",
    )
    # Resampling options
    p.add_argument(
        "--resampling",
        choices=["none", "smote", "smote_tomek", "smote_enn", "under"],
        default="none",
        help="Technique de ré-échantillonnage à utiliser pendant l'entraînement (appliquée uniquement sur les folds/train).",
    )
    p.add_argument(
        "--sampling-strategy",
        default="auto",
        help="Stratégie de sampling imbalanced-learn (ex: 'auto', 0.5, dict). Laissez 'auto' par défaut.",
    )
    p.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=5,
        help="k_neighbors pour SMOTE (utilisé pour smote/smote_tomek/smote_enn).",
    )
    p.add_argument(
        "--balance-per-class",
        type=int,
        default=0,
        help="Si > 0, crée un dataset équilibré avec N échantillons par classe.",
    )
    p.add_argument(
        "--balance-mode",
        choices=["downsample", "upsample", "both"],
        default="both",
        help="Stratégie d’équilibrage: réduire, augmenter, ou les deux.",
    )
    p.add_argument(
        "--balance-classes",
        default=None,
        help="Liste de classes à inclure (séparées par des virgules). Par défaut: toutes.",
    )
    args = p.parse_args()

    print("--- Étape 1: Chargement et Nettoyage des Données ---")
    df = load_any(Path(args.train_file))
    df.columns = df.columns.str.strip()
    
    # Remplacer les valeurs infinies par NaN, qui seront traitées par l'imputer
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"Données chargées: {len(df)} lignes.")

    print("\n--- Étape 2: Préparation des Données ---")
    target_col = args.target or find_target(df)

    # Équilibrage optionnel avant split
    if args.balance_per_class and args.balance_per_class > 0:
        include = None
        if args.balance_classes:
            include = [s.strip() for s in args.balance_classes.split(",") if s.strip()]
        before_counts = df[target_col].value_counts()
        df = balance_df(
            df=df,
            target_col=target_col,
            per_class=args.balance_per_class,
            mode=args.balance_mode,
            seed=args.seed,
            include_classes=include,
        )
        after_counts = df[target_col].value_counts()
        print(f"➡️ Équilibrage appliqué ({args.balance_mode}) à {args.balance_per_class} échantillons par classe.")
        print("Avant:\n", before_counts.to_string())
        print("Après:\n", after_counts.to_string())

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

    def make_training_pipeline():
        """Construit la pipeline d'entraînement, avec resampling si demandé."""
        if args.resampling == "none":
            return Pipeline([("preprocess", preprocess_pipeline), ("model", model)])

        if not HAVE_IMBLEARN:
            raise RuntimeError("imbalanced-learn n'est pas installé mais --resampling est utilisé.")

        # Convert sampling_strategy from CLI to appropriate type (float/int/dict or 'auto')
        sampling_strategy = args.sampling_strategy
        try:
            # try to parse numeric
            if isinstance(sampling_strategy, str) and sampling_strategy != "auto":
                if "." in sampling_strategy:
                    sampling_strategy = float(sampling_strategy)
                else:
                    sampling_strategy = int(sampling_strategy)
        except Exception:
            # keep as string (e.g., 'auto')
            pass

        sampler = None
        if args.resampling == "smote":
            sampler = SMOTE(random_state=args.seed, sampling_strategy=sampling_strategy, k_neighbors=args.smote_k_neighbors)
        elif args.resampling == "smote_tomek":
            sampler = SMOTETomek(random_state=args.seed, sampling_strategy=sampling_strategy, smote=SMOTE(k_neighbors=args.smote_k_neighbors))
        elif args.resampling == "smote_enn":
            sampler = SMOTEENN(random_state=args.seed, sampling_strategy=sampling_strategy, smote=SMOTE(k_neighbors=args.smote_k_neighbors))
        elif args.resampling == "under":
            sampler = RandomUnderSampler(random_state=args.seed, sampling_strategy=sampling_strategy)
        else:
            sampler = None

        if sampler is None:
            return Pipeline([("preprocess", preprocess_pipeline), ("model", model)])
        return ImbPipeline([("preprocess", preprocess_pipeline), ("sampler", sampler), ("model", model)])

    pipe = make_training_pipeline()
    print("Pipeline créé avec succès:")
    print(pipe)

    # Validation croisée (optionnelle)
    if args.cv > 0 and y.nunique() > 1:
        print(f"\n--- Étape 4a: Validation Croisée ({args.cv} folds) ---")
        splits = max(2, min(args.cv, int(y.value_counts().min())))
        if splits < args.cv:
            warnings.warn(f"CV réduite à {splits} folds en raison de classes peu représentées.")
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring=args.scoring, n_jobs=-1)
        print(f"CV {args.scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Folds: {cv_scores}")

    # Évaluation sur un jeu de test (hold-out)
    if y.nunique() > 1:
        print("\n--- Étape 4b: Évaluation sur Jeu de Test (Hold-out) ---")
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=args.testsample, stratify=y, random_state=args.seed)
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

    # Pour l'export, on retire la step de resampling pour éviter toute dépendance
    # à imbalanced-learn en inference et prévenir des problèmes de sérialisation.
    export_pipe = pipe
    if 'ImbPipeline' in globals() and isinstance(pipe, ImbPipeline) and "sampler" in dict(pipe.named_steps):
        export_pipe = Pipeline([
            ("preprocess", pipe.named_steps["preprocess"]),
            ("model", pipe.named_steps["model"]),
        ])

    # Export du modèle
    # dd-mm-yyyy-<model_name>-<resampling>-<score>.skio
    out_file_name = f"{datetime.now().strftime('%d-%m-%Y')}-{args.model}-{args.resampling}-{acc:.4f}.skio"
    out_path = Path(out_file_name)
    if HAVE_SKOPS:
        with open(out_path, "wb") as f:
            skio.dump(export_pipe, f)
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
